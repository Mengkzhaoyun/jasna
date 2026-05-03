# jasna

## git

```bash
git remote add upstream git@github.com:Kruk2/jasna.git
git fetch upstream
git merge v0.6.0
```

## jasna

```powershell
# 触发 release 构建
git checkout release ;`
  git merge main --ff-only ;`
  git push origin release ;`
  git checkout main

# 触发 build 构建
git checkout build ;`
  git merge main --ff-only ;`
  git push origin build ;`
  git checkout main
```

## Local Debugging

```powershell
# 1. 本地测试构建 builder 环境镜像
docker build -f .drone/dockerfile.build -t ghcr.io/mengkzhaoyun/jasna:v0.6.0-alpha5-build --build-arg BASE=nvidia/cuda:13.0.3-devel-ubuntu24.04 .

# 2. 本地测试编译流程 (挂载当前代码并执行 build.sh)
docker pull ghcr.io/mengkzhaoyun/jasna:v0.6.0-alpha5-build ; `
docker run --rm -it `
  -v "${PWD}:/app/jasna" `
  -w /app/jasna `
  ghcr.io/mengkzhaoyun/jasna:v0.6.0-alpha5-build `
  bash .drone/build.sh
```