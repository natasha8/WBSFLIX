# 🎥🍿[Go WBSFLIX Movie Recommendation System](https://gowbsflix.streamlit.app/) 🎥🍿

Go WBSFLIX is a movie recommendation system built with Streamlit, Pandas, scikit-learn, and the IMDbPY package. It provides various types of movie recommendations based on user preferences, movie ratings, similarity, and tags.

## Features

- **Top Movies**: Recommends movies based on high average ratings and count of ratings.
- **Similar Movies by Titles**: Finds movies similar to a given movie title.
- **User Recommendations**: Offers recommendations based on a specific user's rating history.
- **Tag-Based Movies**: Suggests movies based on specified tags or moods.

## Installation

To run Go WBSFLIX on your local machine, follow these steps:

### Prerequisites

- Python 3.6 or above
- pip (Python package installer)

### Setup

1. **Clone the Repository**

   ```bash
   git clone https://github.com/natasha8/WBSFLIX.git
   cd WBSFLIX
   ```

2. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Application**

   ```bash
   streamlit run app.py
   ```

   The application should now be running on `http://localhost:8501`.

## Usage

After launching Go WBSFLIX, navigate through the sidebar to choose your recommendation type:

1. **Top Movies**: Select a range of years and the number of top movies to display.
2. **Similar Movies by Titles**: Enter a movie title to find similar movies.
3. **User Recommendations**: Provide a user ID to get personalized recommendations.
4. **Tag-Based Movies**: Enter a mood or tag to get relevant movie suggestions.

## Data Sources

The recommendation system uses the following datasets:

- MovieLens Dataset for ratings and movie metadata.
- IMDbPY for fetching movie posters and additional details.

## Contributing

Contributions to Go WBSFLIX are welcome! Please read `CONTRIBUTING.md` for details on our code of conduct, and the process for submitting pull requests to us.

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.

## Acknowledgments

- MovieLens for providing the dataset.
- IMDbPY for their API to fetch movie details.
- Streamlit for their easy-to-use app development framework.
