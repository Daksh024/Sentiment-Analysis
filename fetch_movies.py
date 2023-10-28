import requests
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
from collections import Counter
import tkinter as tk
from tkinter import ttk, Label
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

review_dict = {}

def fetch_all_movie_names():
    imdb_url = "https://www.imdb.com/list/ls565461384/"
    try:
        response = requests.get(imdb_url)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')
        movies = soup.find_all('h3', class_='lister-item-header')
        for movie in movies:
            title = movie.a.text
            link = "https://www.imdb.com" + movie.a['href'] + 'reviews/'
            review_dict.update({
                title:link
            })

        movie_list.delete(0, tk.END)
        for movie_name in review_dict.keys():
            movie_list.insert(tk.END, movie_name)

    except requests.RequestException as e:
        show_error("Error fetching movie names: " + str(e))

def enable_analysis_button(event):
    selected_movie = movie_list.get(movie_list.curselection())
    analysis_button.config(state=tk.NORMAL)
    analysis_button.config(command=lambda: analyze_movie(selected_movie))

def analyze_movie(movie_name):
    print(movie_name)
    print("Analysing Movie")
    import main
    import pickle

    import pandas as pd
    
    import seaborn as sns  

    # from train import model_data

    # le, tfidf = model_data()

    le = pickle.load(open('le.pickle','rb'))
    classifier = pickle.load(open('clf.sav', 'rb'))
    print('='*50,'model loaded','='*50)

    tfidf = pickle.load(open('tfidf.pickle', 'rb'))

    movie_review_df = pd.DataFrame(main.get_reviews(
        start_url   = review_dict[movie_name],
        link        = review_dict[movie_name]+'_ajax/'))

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

def get_movie_genres():
    try:
        url = "https://www.imdb.com/search/title/?languages=hi&year=2023,2023"
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")

        genres = []
        for item in soup.find_all("div", class_="lister-item-content"):
            genre_tag = item.find("span", class_="genre")
            if genre_tag:
                genre_list = genre_tag.text.strip().split(',')
                genres.extend([g.strip() for g in genre_list])

        return Counter(genres)

    except requests.RequestException as e:
        show_error("Connection Error: " + str(e))
        return Counter()

def show_error(message):
    error_label = Label(main_window, text=message, fg="red")
    error_label.pack(pady=10)
    main_window.after(5000, error_label.destroy)

def plot_genre_graph():
    genre_count = get_movie_genres()
    if not genre_count:
        return

    genres = list(genre_count.keys())
    counts = list(genre_count.values())

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(genres, counts, color='blue')
    ax.set_ylabel('Number of Movies')
    ax.set_xlabel('Genre')
    ax.set_title('2023 Hindi Movies by Genre on IMDb')
    plt.xticks(rotation=45)

    canvas = FigureCanvasTkAgg(fig, master=main_window)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

main_window = tk.Tk()
main_window.title("Indian Movies from 2023")
main_window.geometry("600x400")

frame = tk.Frame(main_window)
frame.pack(side=tk.LEFT, fill=tk.Y)

movie_list = tk.Listbox(frame, width=50)
movie_list.pack(fill=tk.BOTH, expand=True)
movie_list.bind("<<ListboxSelect>>", enable_analysis_button)

fetch_button = tk.Button(main_window, text="Fetch All Movie Names", command=fetch_all_movie_names)
fetch_button.pack()

analysis_button = tk.Button(main_window, text="Analyze", state=tk.DISABLED)
analysis_button.pack()

genre_graph_button = tk.Button(main_window, text="Plot Genre Graph", command=plot_genre_graph)
genre_graph_button.pack()

main_window.mainloop()
