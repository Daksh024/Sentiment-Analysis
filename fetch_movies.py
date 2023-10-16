import tkinter as tk
import requests
from bs4 import BeautifulSoup

review_dict = {}

# Function to fetch and display all movie names from 2023
def fetch_all_movie_names():
    imdb_url = "https://www.imdb.com/list/ls565461384/"
    response = requests.get(imdb_url)

    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')

        # Find all the movie titles and links
        movies = soup.find_all('h3', class_='lister-item-header')

        # Extract and print the titles and links
        for movie in movies:
            title = movie.a.text
            link = "https://www.imdb.com" + movie.a['href'] + 'reviews/'
            review_dict.update({
                title:link
            })

        movie_list.delete(0, tk.END)  # Clear the current movie list
        for movie_name in review_dict.keys():
            movie_list.insert(tk.END, movie_name)

# Function to enable analysis button when a movie is selected
def enable_analysis_button(event):
    selected_movie = movie_list.get(movie_list.curselection())
    analysis_button.config(state=tk.NORMAL)
    analysis_button.config(command=lambda: analyze_movie(selected_movie))

# Function to perform analysis for the selected movie
def analyze_movie(movie_name):
    print(movie_name)
    print("Analysing Movie")
    import main
    import pickle

    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure    

    from train import model_data

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

    # Create a Tkinter canvas for Matplotlib plot

    # analysis_window = tk.Toplevel(main_window)
    # canvas = FigureCanvasTkAgg(fig, master=analysis_window)
    # canvas_widget = canvas.get_tk_widget()
    # canvas_widget.pack()


    # analysis_window.title(f"Analysis for {movie_name}")
    # analysis_label = tk.Label(analysis_window, text=f"Perform analysis for {movie_name}")
    # analysis_label.pack()

# Create the main tkinter window with a custom geometry
main_window = tk.Tk()
main_window.title("Movies from 2023")
main_window.geometry("600x400")  # Set the window size (width x height)

# Create a frame to hold the listbox and place it on the left
frame = tk.Frame(main_window)
frame.pack(side=tk.LEFT, fill=tk.Y)

# Create a listbox to display movie names
movie_list = tk.Listbox(frame,width=50)
movie_list.pack(fill=tk.BOTH, expand=True)
movie_list.bind("<<ListboxSelect>>", enable_analysis_button)  # Bind the selection event

# Create a button to fetch and display all movie names from 2023
fetch_button = tk.Button(main_window, text="Fetch All Movie Names", command=fetch_all_movie_names)
fetch_button.pack()

# Create a button for analysis (initially disabled)
analysis_button = tk.Button(main_window, text="Analyze", state=tk.DISABLED)
analysis_button.pack()

# Start the tkinter main loop
main_window.mainloop()
