# Publishing the repository and obtaining the DOI

CNSNS asks for research data to be deposited in a repository, cited and linked in the article (Option C of the Elsevier research-data policy). Do this once, in this order.

**Names to use**
* GitHub repository name: `descriptor-residual-learning-mpc` (public, MIT licence, no extra README or licence added on the GitHub page)
* Release tag: `v1.0.0`; release title: `Structure-preserving residual learning and predictive control of nonlinear descriptor power-network models, version 1.0.0`
* Zenodo record title (taken from `.zenodo.json`): `Code and data for: Structure-preserving residual learning and predictive control of nonlinear descriptor power-network models`

**Steps**
1. On GitHub choose *New repository*, name `descriptor-residual-learning-mpc`, set it to *Public*, and create it empty.
2. In a terminal inside this folder: `git init`, `git add .`, `git commit -m "Initial release"`, `git branch -M main`, `git remote add origin https://github.com/<your-user>/descriptor-residual-learning-mpc.git`, `git push -u origin main`.
3. In `CITATION.cff` and `.zenodo.json` replace `YOUR-GITHUB-USERNAME` by your GitHub user name, commit and push.
4. Log in at https://zenodo.org with your GitHub account, open *Account > GitHub*, find `descriptor-residual-learning-mpc` and switch it on.
5. On GitHub open *Releases > Draft a new release*, create the tag `v1.0.0` on `main`, enter the release title above, and publish. Zenodo archives it within a few minutes and mints the DOI (shown on the Zenodo record, form `10.5281/zenodo.NNNNNNN`).
6. Open the Zenodo record, check title, author, ORCID, licence and version, and copy the DOI.
7. Send the GitHub URL and the DOI so that they can be inserted into the manuscript (`[repository DOI to be inserted before submission]`), `Declarations.docx`, `Cover_letter.docx` and `CITATION.cff`, and paste the DOI into the Editorial Manager data-linking field. Open the DOI in a private browser window to confirm that it resolves before you submit.

Alternative without GitHub: upload the zip at https://zenodo.org/uploads/new (type *Software*), publish it, and use the DOI it gives you.
