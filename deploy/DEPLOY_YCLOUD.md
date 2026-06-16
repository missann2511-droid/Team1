# Deploying to Yandex Cloud (W8.1 + W8.2)

This guide deploys the backend (FastAPI) and frontend (Streamlit) as
**Serverless Containers** behind public URLs, and wires up **autodeploy**
via the `deploy.yml` GitHub Actions workflow.

You can use Serverless Containers (simplest, pay-per-use) or a Compute VM
running `docker compose`. Steps below cover the Serverless Container path.

---

## 0. Prerequisites

- A Yandex Cloud account with billing enabled.
- The `yc` CLI installed locally: <https://yandex.cloud/docs/cli/quickstart>
- Docker installed locally (only needed for the manual first push).

Set convenience variables:

```bash
yc init                       # log in, pick cloud + folder
export FOLDER_ID=$(yc config get folder-id)
export CLOUD_ID=$(yc config get cloud-id)
```

---

## 1. Create a Container Registry

```bash
yc container registry create --name vehicle-price
export REGISTRY_ID=$(yc container registry get vehicle-price --format json | jq -r .id)
yc container registry configure-docker
```

## 2. Create a service account for CI/CD

```bash
yc iam service-account create --name ci-deployer
export SA_ID=$(yc iam service-account get ci-deployer --format json | jq -r .id)

# Roles: push images, deploy containers, let containers pull images
for ROLE in container-registry.images.pusher serverless.containers.editor container-registry.images.puller; do
  yc resource-manager folder add-access-binding $FOLDER_ID \
    --role $ROLE --subject serviceAccount:$SA_ID
done

# Authorized key (store the file content as the YC_SA_KEY_JSON secret)
yc iam key create --service-account-id $SA_ID --output sa-key.json
```

## 3. Create the two Serverless Containers

```bash
yc serverless container create --name vehicle-backend
yc serverless container create --name vehicle-frontend
export BACKEND_CONTAINER_ID=$(yc serverless container get vehicle-backend --format json | jq -r .id)
export FRONTEND_CONTAINER_ID=$(yc serverless container get vehicle-frontend --format json | jq -r .id)
```

## 4. First manual deploy (bootstrap)

```bash
# Train + select best model so the image ships with models/best/model_pipeline.joblib
python -m src.preprocess
for m in ridge random_forest gradient_boosting knn; do python -m src.train --model $m; done
python -m src.select_best

REGISTRY=cr.yandex/$REGISTRY_ID

docker build -t $REGISTRY/backend:bootstrap -f backend/Dockerfile .
docker push $REGISTRY/backend:bootstrap
yc serverless container revision deploy \
  --container-id $BACKEND_CONTAINER_ID \
  --image $REGISTRY/backend:bootstrap \
  --cores 1 --memory 1GB --concurrency 4 --execution-timeout 30s \
  --service-account-id $SA_ID

# Make the backend public and capture its URL
yc serverless container allow-unauthenticated-invoke vehicle-backend
export BACKEND_URL=$(yc serverless container get vehicle-backend --format json | jq -r .url)

docker build -t $REGISTRY/frontend:bootstrap -f frontend/Dockerfile .
docker push $REGISTRY/frontend:bootstrap
yc serverless container revision deploy \
  --container-id $FRONTEND_CONTAINER_ID \
  --image $REGISTRY/frontend:bootstrap \
  --cores 1 --memory 512MB --concurrency 8 --execution-timeout 30s \
  --environment BACKEND_URL=$BACKEND_URL \
  --service-account-id $SA_ID
yc serverless container allow-unauthenticated-invoke vehicle-frontend
```

Open the frontend URL:

```bash
yc serverless container get vehicle-frontend --format json | jq -r .url
```

Verify the backend:

```bash
curl $BACKEND_URL/health
```

---

## 5. Enable autodeploy (W8.2)

Add these **GitHub repository secrets** (Settings -> Secrets and variables -> Actions):

| Secret | Value |
|--------|-------|
| `YC_SA_KEY_JSON` | full contents of `sa-key.json` |
| `YC_CLOUD_ID` | `$CLOUD_ID` |
| `YC_FOLDER_ID` | `$FOLDER_ID` |
| `YC_REGISTRY_ID` | `$REGISTRY_ID` |
| `YC_BACKEND_CONTAINER_ID` | `$BACKEND_CONTAINER_ID` |
| `YC_FRONTEND_CONTAINER_ID` | `$FRONTEND_CONTAINER_ID` |
| `YC_SA_ID` | `$SA_ID` |
| `YC_BACKEND_PUBLIC_URL` | the backend URL from step 4 |

Now every push to `main` runs `.github/workflows/deploy.yml`, which rebuilds
the images, pushes them tagged with the commit SHA, and deploys a new revision
of each container. Roll back any time with:

```bash
yc serverless container revision list --container-name vehicle-backend
yc serverless container revision activate <REVISION_ID>
```

---

## Alternative: Compute VM + docker compose

1. `yc compute instance create ... --create-boot-disk ... ` (Ubuntu, public IP).
2. SSH in, install Docker + the Compose plugin.
3. `git clone` the repo, run the model pipeline once, then `docker compose up -d`.
4. Open ports 8000 (API) and 8501 (UI) in the security group.
5. For autodeploy, replace the deploy steps in `deploy.yml` with an SSH action
   that runs `git pull && docker compose up -d --build` on the VM.
