# Rotten Cabbages
*Group members: Rami Gorovoi-Abu Hashish, Xintong Gu, Ren Liao, Mariia Onokhina, Charles Xiong*

This is an in-progress movie recommendation system built on public datasets and various filters for the **Fall 2025 NYU Tandon Data Science Bootcamp**. We aim to provide personalized and dynamic movie discovery.

**Datasets used:**
- TMDB 5000 Movie Dataset [https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata]
- MovieLens Dataset [https://grouplens.org/datasets/movielens/]

**Recommendation models used:**
- Demographic filtering
- Collaborative filtering
- Content-based filtering

Our goal is to finalize the content-based filter and create a functional web application to implement these reccomendation models.

## Instructions to Run Locally

**Clone the repository:**

```bash
git clone https://github.com/RGorovoi/Rotten-Cabbages.git
```

**Move inside the root folder:**

```bash
cd Rotten-Cabbages
```

**If you do not have Fast API installed:**

```bash
pip install "fastapi[standard]"
```

**Start Fast API server on port 8000:**

```bash
cd API
uvicorn main:app --reload --port 8000
```

**To check that it works, open http://127.0.0.1:8000/health in your browser.** 

If it shows, {"ok": true}, the server is working properly.

**Next, move into the frontend folder:**

```bash
cd ../rotten-cabbages
```

**If you do not have HomeBrew installed, install it and then install Node.js:**

```bash
curl -o- https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh | bash
brew install node@24

```

You may also use another package manager. Visit this link: https://nodejs.org/en/download

**Inside of rotten-cabbages folder, type the following:**

```bash
npm install
npm run dev
```

This should open a new tab running the website locally. Check the terminal messages for a link to the locally hosted website. 

If it doesn't exist, you are probably missing a package.

Missing packages can be installed with 

```bash
npm install <package_name>
```

### List of Packages / Libraries Used

```bash
npm install react-router-dom
npm install axios
```
