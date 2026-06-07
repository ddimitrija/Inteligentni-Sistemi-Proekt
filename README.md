# Inteligentni-Sistemi-Proekt
Предлог за Проект (Project Proposal)




1. Учесници : Димитрија Ѓошевски

2. Наслов на проектот : Music Taste Discovery Agent (Spotify)

3. Тип на проект : Clustering & Machine Learning

4. Опис на проектот : This project creates an intelligent music recommendation agent that analyzes Spotify songs using clustering and machine learning. It groups songs based on audio features like tempo, energy, and danceability to detect listening patterns. The agent is designed for Spotify users who want personalized music suggestions tailored to their tastes. Based on the identified clusters, it recommends new songs similar to the user's preferences while allowing some exploration of new styles. This system demonstrates how machine learning can uncover patterns in music and support intelligent, adaptive recommendations.

5. AI API(и) (ако е применливо) : No AI/APIs used.

6. Датасет (ако е применливо) : A reference playlist with 1000+ songs is used.

7. Карактеристики на проектот : Analyze music features, cluster songs, recommend songs.

8. Очекуван резултат
Example :
Recommended songs :
Song A - Artist X
Song B - Artist Y
Song C - Artist Z

## Running the Project

**Install dependencies:**
```bash
pip install -r requirements.txt
```

**Run with your own playlist:**

Export any Spotify playlist as a CSV from [Exportify](https://exportify.net), place it in `spotify/`, then:
```bash
python main.py --source spotify/yourplaylist.csv
```

On first run the reference cache will be built automatically — this takes ~10 seconds and only happens once.

**Output files are written to the project folder:**
- `recommendation_report.html` — open in any browser
- `recommendations.csv` — flat table of recommended tracks
