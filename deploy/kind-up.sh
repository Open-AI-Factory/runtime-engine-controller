#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
IMG=grp-runtime-cpu:0.1

docker image inspect $IMG >/dev/null 2>&1 || docker build -t $IMG -f deploy/Dockerfile.cpu deploy/
workers=$(kind get nodes --name grp 2>/dev/null | grep -c worker || true)
if [ "$workers" != "2" ]; then
  kind delete cluster --name grp >/dev/null 2>&1 || true
  sed "s#__REPO_DIR__#$PWD#g" deploy/kind-cluster.yaml > /tmp/grp-kind-cluster.yaml
  kind create cluster --name grp --config /tmp/grp-kind-cluster.yaml
fi
for n in $(kind get nodes --name grp); do docker start "$n" >/dev/null; done
kubectl wait --for=condition=Ready node --all --timeout=180s >/dev/null
kind load docker-image $IMG --name grp
kubectl apply -f deploy/crd-runtimeaction.yaml -f deploy/crd-executionplan.yaml -f deploy/crd-planrequest.yaml -f deploy/rbac.yaml
kubectl get nodes
