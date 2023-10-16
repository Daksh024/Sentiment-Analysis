import requests
from bs4 import BeautifulSoup

import train 

def get_reviews(start_url='',link=''):

    reviews = []

    # start_url = 'https://www.imdb.com/title/tt15354916/reviews?ref_=tt_urv'
    # link = 'https://www.imdb.com/title/tt15354916/reviews/_ajax'

    params = {
        'ref_': 'undefined',
        'paginationKey': ''
    }

    with requests.Session() as s:
        s.headers['User-Agent'] = 'Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.150 Safari/537.36'
        res = s.get(start_url)

        while True:
            soup = BeautifulSoup(res.text,"lxml")
            for item in soup.select(".review-container"):
                reviewer_name = item.select_one("span.display-name-link > a").get_text(strip=True)
                review_content = item.select_one("div.text.show-more__control").get_text(strip=True)
                review_date = item.select_one("span.review-date").get_text(strip=True)
                
                reviews.append({
                    "review":review_content,
                    "date":review_date
                })

            try:
                pagination_key = soup.select_one(".load-more-data[data-key]").get("data-key")
            except AttributeError:
                break
            params['paginationKey'] = pagination_key
            res = s.get(link,params=params)

    return reviews
#--------------------------------------------------------------------

if __name__ == "__main__":

    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns


    le, tfidf = train.model_data()

    import pickle
    # load the model from disk
    classifier = pickle.load(open('clf.sav', 'rb'))
    # result = loaded_model.score(X_test, Y_test)
    print('='*50,'model loaded','='*50)

    tfidf = pickle.load(open('tfidf.pickle', 'rb'))

    movie_review_df = pd.DataFrame(get_reviews())

    movie_review_df['date'] = pd.to_datetime(movie_review_df['date'])

    movie_review_df.sort_values(by='date',inplace=True,ignore_index=True)

    vectorize_reviews = tfidf.transform(movie_review_df['review'])
    prediction = classifier.predict(vectorize_reviews)
    movie_review_df['Prediction'] = le.inverse_transform(prediction)


    sum_ = []
    cnt = 0
    for i in prediction:
        f = lambda x: -1 if x == 0 else 1
        cnt += f(i)
        sum_.append(cnt)

    movie_review_df['Popularity'] = sum_

    import matplotlib.dates as mdates

    fig, ax = plt.subplots(figsize=(10, 6))  # set the size of the figure
    sns.lineplot(x='date', y='Popularity', data=movie_review_df)

    # Formatting the x-axis to show dates in the format "YYYY-MM-DD"
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m'))
    plt.xticks(rotation=45)  # rotate x-axis labels for better readability

    # Display the grid
    ax.grid(True)

    plt.title("Popularity of Movies")
    plt.show()

    # Calculate the frequency of each unique value in the "Prediction" column
    prediction_counts = movie_review_df['Prediction'].value_counts()

    # Plotting the pie chart
    plt.figure(figsize=(8, 6))
    plt.pie(prediction_counts, labels=prediction_counts.index, autopct='%1.1f%%', startangle=140, wedgeprops=dict(width=0.3))
    plt.title('Prediction Distribution')
    plt.show()
