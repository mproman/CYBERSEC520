# Publishing to GitHub Pages

There are two main ways to publish your Jupyter Book: using **GitHub Actions** (recommended for automation) or manually using the `ghp-import` tool.

## Option 1: GitHub Actions (Recommended)

This method automatically builds and deploys your book whenever you push changes to the `main` branch.

1.  **Create the Workflow File**:
    Create a file in your repository at `.github/workflows/deploy.yml` with the following content:

    ```yaml
    name: deploy-book

    # Run on pushes to the main branch
    on:
      push:
        branches:
          - main

    # Sets permissions of the GITHUB_TOKEN to allow deployment to GitHub Pages
    permissions:
      contents: read
      pages: write
      id-token: write

    # Allow one concurrent deployment
    concurrency:
      group: "pages"
      cancel-in-progress: true

    jobs:
      deploy-book:
        runs-on: ubuntu-latest
        steps:
          - uses: actions/checkout@v4

          - name: Install uv
            uses: astral-sh/setup-uv@v1

          - name: Set up Python
            run: uv python install 3.11

          - name: Install dependencies
            run: uv pip install -r requirements.txt

          - name: Build the book
            run: uv run jupyter-book build .

          - name: Upload artifact
            uses: actions/upload-pages-artifact@v3
            with:
              path: "_build/html"

          - name: Deploy to GitHub Pages
            id: deployment
            uses: actions/deploy-pages@v4
    ```

2.  **Configure GitHub Settings**:
    *   Go to your repository on GitHub.
    *   Click **Settings** > **Pages**.
    *   Under **Build and deployment**, select **GitHub Actions** as the source.

## Option 2: Manual Publish (ghp-import)

Use this method if you want to deploy from your local machine manually.

1.  **Install ghp-import**:
    ```bash
    uv pip install ghp-import
    ```

2.  **Build and Deploy**:
    Run these commands from the root of your project:
    ```bash
    # 1. Build the book (if you haven't already)
    uv run jupyter-book build .

    # 2. Push the _build/html folder to the 'gh-pages' branch
    uv run ghp-import -n -p -f _build/html
    ```

3.  **Configure GitHub Settings**:
    *   Go to your repository on GitHub.
    *   Click **Settings** > **Pages**.
    *   Under **Build and deployment**, select **Deploy from a branch**.
    *   Select `gh-pages` as the branch and `/ (root)` as the folder.
