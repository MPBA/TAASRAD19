Code for an interactive [UMAP](https://github.com/lmcinnes/umap) visualization of a subset of the TAASRAD19 data set.
Demo at [http://taasrad19.westus2.cloudapp.azure.com:3000/](http://taasrad19.westus2.cloudapp.azure.com:3000/).
You can read more about the demo in [the about section](http://taasrad19.westus2.cloudapp.azure.com:3000/#about).

### Original Code

This demo is based on the Umap explorer tool developed by [GrantCuster](https://github.com/GrantCuster).
The original version of the code can be found at [https://github.com/GrantCuster/umap-explorer](https://github.com/GrantCuster/umap-explorer)

## A rough guide to the code

The demo app is a React app. It uses a `src/Data.js` to fetch the data and `src/Layout.js` to handle the layout of the page. The three.js visualization code is in `src/Projection.js`. The texture atlases are in the public folder as images. 

The following iPython notebooks are included:
1. `01_compute_umap.ipynb` contains the code for computing UMAP on th e TAASRAD19 dataset and saving the generated embeddings.
2. `02_umap_make_images.ipynb` generates the texture atlases that are then shown in the interactive visualization.

## Running the app

This project was bootstrapped with [Create React App](https://github.com/facebook/create-react-app).

In the project directory, you can run:

### `npm start`

Runs the app in the development mode.<br>
Open [http://localhost:3000](http://localhost:3000) to view it in the browser.

The page will reload if you make edits.<br>
You will also see any lint errors in the console.

### `npm run build`

Builds the app for production to the `build` folder.<br>
It correctly bundles React in production mode and optimizes the build for the best performance.

The build is minified and the filenames include the hashes.<br>
Your app is ready to be deployed!

See the section about [deployment](https://facebook.github.io/create-react-app/docs/deployment) for more information.

### Learn more about create react app

You can learn more in the [Create React App documentation](https://facebook.github.io/create-react-app/docs/getting-started).

To learn React, check out the [React documentation](https://reactjs.org/).

