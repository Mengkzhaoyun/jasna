# jasna

## git

```bash
git remote add upstream git@github.com:Kruk2/jasna.git
git fetch upstream
git merge v0.6.0-alpha5
```

## jasna

```powershell
# 触发 v0.6.0-alpha5 构建
git checkout v0.6.0-alpha5 ;`
  git merge main --ff-only ;`
  git push origin v0.6.0-alpha5 ;`
  git checkout main

# 触发 build 构建
git checkout build ;`
  git merge main --ff-only ;`
  git push origin build ;`
  git checkout main
```