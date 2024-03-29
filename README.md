# slide

This is my slide repository. Powered by [slidev](https://sli.dev/).

## Usage

You need to install [Node.js](https://nodejs.org/en/) first, which requires Node.js `14.0.0` or higher.

Copy the `test` folder to your own folder, and run the following command:

```bash
npm install
npm install -g @slidev/cli
npm run dev
```

Then you can visit http://localhost:3030 to see the slides.

Change the `slides.md` to see the changes. For more information, please visit [documentations](https://sli.dev/).

## Export

```bash
npm i -D playwright-chromium
slidev export
slidev export --dark
slidev export-notes
```

Then you can find the exported PDF file in the `dist` folder.

## Deploy

For GitHub Pages, you need to provide the permission for GitHub Actions to deploy the slides.

`Settings` -> `Actions` -> `General` -> `Workflow permissions` -> `Read and write permissions` -> `Save`

Then you can push the code to the `main` branch, and the slides will be deployed to the `gh-pages` branch.

## License

CC0-1.0