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