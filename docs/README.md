# mllabiome documentation

The documentation site is built with Next.js and Nextra and exported as static files for GitHub Pages.

## Local development

```bash
cd docs
npm ci
npm run dev
```

The local development server uses port 8888.

## Production build

```bash
cd docs
npm ci
npm run build
```

The static site is written to `docs/out/`.

To emulate the GitHub project-site path locally during a build:

```bash
BASE_PATH=/mllabiome NEXT_PUBLIC_BASE_PATH=/mllabiome npm run build
```

## GitHub Pages

The repository workflow `.github/workflows/docs.yml` builds `docs/` and deploys `docs/out/` with GitHub Pages Actions.

In GitHub, open **Settings → Pages → Build and deployment** and set **Source** to **GitHub Actions**. Pushes to `main` that change the documentation or workflow will then publish the site.
