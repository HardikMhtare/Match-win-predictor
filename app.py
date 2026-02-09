from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load the trained model
pipe = pickle.load(open('pipe.pkl', 'rb'))

teams = ['Royal Challengers Bangalore', 'Mumbai Indians', 'Kings XI Punjab', 'Kolkata Knight Riders',
         'Sunrisers Hyderabad', 'Rajasthan Royals', 'Chennai Super Kings', 'Deccan Chargers', 'Delhi Capitals']

city = ['Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi', 'Chandigarh', 'Jaipur', 'Chennai',
        'Cape Town', 'Port Elizabeth', 'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
        'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala', 'Visakhapatnam', 'Pune', 'Raipur',
        'Ranchi', 'Abu Dhabi', 'Sharjah', 'Mohali', 'Bengaluru']

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        # Collect form data
        batting_team = request.form.get('batting_team')
        bowling_team = request.form.get('bowling_team')
        selected_city = request.form.get('city')
        target = int(request.form.get('target'))
        score = int(request.form.get('score'))
        overs = float(request.form.get('overs'))
        wickets = int(request.form.get('wickets'))

        # Calculate input parameters
        runs_left = target - score
        balls_left = 120 - (overs * 6)
        wickets_left = 10 - wickets
        crr = score / overs if overs > 0 else 0  # Avoid division by zero
        rrr = (runs_left * 6) / balls_left if balls_left > 0 else 0  # Avoid division by zero

        # Create dataframe for prediction
        input_df = pd.DataFrame({
            'batting_team': [batting_team],
            'bowling_team': [bowling_team],
            'city': [selected_city],
            'runs_left': [runs_left],
            'balls_left': [balls_left],
            'wickets_left': [wickets_left],
            'total_runs_x': [target],
            'crr': [crr],
            'rrr': [rrr]
        })

        # Predict the win probability
        result = pipe.predict_proba(input_df)
        loss = result[0][0]
        win = result[0][1]

        # Format the prediction
        prediction = {
            "batting_team": f"{batting_team}: {round(win * 100)}%",
            "bowling_team": f"{bowling_team}: {round(loss * 100)}%"
        }

    return render_template('index.html', teams=teams, cities=city, prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
