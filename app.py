import os, pickle, requests
from flask import Flask, render_template, jsonify, request
from newsapi import NewsApiClient

app = Flask(__name__, template_folder="templates")

# Load ML model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Load NewsAPI key
newsapi = NewsApiClient("e259e113209242a99fff873b6749d2c6")

SAMPLE_NEWS = [
    {"title": "India launches new satellite for communication", "description": "ISRO successfully launched GSAT-30.", "url": "#"},
    {"title": "Cricket World Cup 2025: India defeats Australia", "description": "Massive win by India in the finals.", "url": "#"},
    {"title": "Global economy stabilizes post-pandemic", "description": "IMF projects 3.2% global growth this year.", "url": "#"},
    {"title": "AI helping detect fake news in media", "description": "New tools enhance online fact-checking.", "url": "#"}
]



def predict_news(text):
    """Predict if a piece of news is fake or real."""
    try:
        vec = vectorizer.transform([text])
        pred = model.predict(vec)[0]
        proba = model.predict_proba(vec)[0]
        real_prob = round(proba[1] * 100, 2)
        fake_prob = round(proba[0] * 100, 2)
        label = "Likely Real" if pred == "REAL" else "Likely Fake"
        return {"label": label, "real_prob": real_prob, "fake_prob": fake_prob}
    except Exception as e:
        print("Prediction error:", e)
        return {"label": "N/A", "real_prob": 0, "fake_prob": 0}


def format_articles(articles):
    """Format and predict for all articles."""
    formatted = []
    for a in articles:
        title = a.get("title", "")
        desc = a.get("description", "")
        text = f"{title} {desc}"
        prediction = predict_news(text)
        formatted.append({
            "title": title or "Untitled",
            "description": desc or "",
            "url": a.get("url", "#"),
            "urlToImage": a.get("urlToImage") or "https://via.placeholder.com/300x200?text=No+Image",
            **prediction
        })
    return formatted


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/fetch_all")
def fetch_all():
    """Fetch Indian, Top, Global, Sports news (with fallback)."""
    try:
        data = {}

        # 🇮🇳 Indian News (with fallback)
        indian = newsapi.get_top_headlines(country="in", page_size=6).get("articles", [])
        if not indian:
            print("Indian news empty → fallback search")
            indian = newsapi.get_everything(q="India", language="en", page_size=6).get("articles", [])
        data["indian"] = format_articles(indian)

        # 🏆 Top 10 (US + general)
        top = newsapi.get_top_headlines(country="us", page_size=6).get("articles", [])
        if not top:
            top = newsapi.get_everything(q="Top news", language="en", page_size=6).get("articles", [])
        data["top"] = format_articles(top)

        # 🌍 Global News
        global_news = newsapi.get_everything(q="world news OR international", language="en", page_size=6).get("articles", [])
        data["global"] = format_articles(global_news)

        # 🏏 Sports (with fallback)
        sports = newsapi.get_top_headlines(category="sports", country="in", page_size=6).get("articles", [])
        if not sports:
            print("Sports news empty → fallback search")
            sports = newsapi.get_everything(q="sports cricket football", language="en", page_size=6).get("articles", [])
        data["sports"] = format_articles(sports)

        print("✅ News fetched successfully.")
        return jsonify(data)

    except Exception as e:
        print("Error fetching news:", e)
        return jsonify({"error": str(e)})


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("news", "")
    return jsonify(predict_news(text))


if __name__ == "__main__":
    app.run(debug=True)
