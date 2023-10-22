
import streamlit as st
import pickle as pk
import pandas as pd

def predict_system(user_input):
    if user_input is not None:
        with open('scaler_model.pkl', 'rb') as scaler_file:
            scaler = pk.load(scaler_file)
     
        user_inputs_scaled = scaler.transform(user_input)


        try:
            with open('best_model.pkl', 'rb') as file:
                loaded_model = pk.load(file)
                prediction = loaded_model.predict(user_inputs_scaled)[0]
                st.success(f"Predicted Rating: {prediction}")
                return prediction
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
            return None
    else:
        st.warning("Please fill in all fields.")
        return None




st.set_page_config(page_title="Predict a Player's Rating", page_icon="⚽", layout="wide")

st.subheader("Hey Soccer fan, Welcome 😃")
st.title("Predict a Player's rating")
st.write("Here, you can enter some features of a player and we wil predict their ratings!")

with st.container():
    st.write("---")
    left_column, right_column = st.columns(2)
    with left_column:
        st.header("Guidelines")
        st.write("##")
        st.write("""
    Please enter the following features of the player.  
    Apart from Value, Wage and Release clause, all other features are in percentages
  1. Potential: Player's potential skill level or overall capability, often used to project how much a player can improve.

  2. Value in euro: Market value of the player in euros, reflecting their perceived worth in the transfer market.

  3. Wage in euro: Player's weekly or monthly salary, denominated in euros.

  4. Age: Player's age

  5. International Reputation: A rate of the player's international reputation.(between 0 and 5 inclusive)  

  6. Release clause eur: The release clause is the amount of money that another club needs to pay to buy out a player's contract. This is stated in euros.
                 
  7. Shooting: Player's ability to make accurate shots

  8. Passing: Player's ability to make accurate and effective passes.

  9. Dribbling: Indicates a player's skill in maneuvering the ball while running.
                   
  10. Physic: The player's overall well-being and health

  11. Attacking : This is a subset of passing, specifically focusing on short passes used in attacking play.

  12. Movement reactions: Refers to a player's ability to react quickly and move effectively on the field.

  13. Movement agression: The intensity with which the player moves

  14. Power shot power: Relates to a player's ability to take powerful shots on goal.

 15. Mentality vision: Reflects a player's strategic vision on the field, understanding the game and making intelligent decisions.

 16. Mentality composure: Indicates how composed and focused a player is, especially in high-pressure situations.

 17. Power average: An aggregate measure of a player's physical power across different attributes.
""")
        
    with right_column:
        st.header("Predict!")
        st.write("##")
        # ------------------------------------------------------------------------ #

        st.subheader("Enter Player's Information:")
        # Get user input for each variable
        potential = st.number_input("Potential", min_value=0, max_value=100, step=1)
        value_eur = st.number_input("Value (in Euros)", min_value=0.0, step=100.0)
        wage_eur = st.number_input("Wage (in Euros)", min_value=0.0, step=100.0)
        age = st.number_input("Age", min_value=0, step=1)
        international_reputation = st.number_input("International Reputation", min_value=0, max_value=5, step=1)
        release_clause_eur = st.number_input("Release Clause (in Euros)", min_value=0.0, step=100.0)
        shooting = st.number_input("Shooting", min_value=0, max_value=100, step=10)
        passing = st.number_input("Passing", min_value=0, max_value=100, step=1)
        dribbling = st.number_input("Dribbling", min_value=0, max_value=100, step=1)
        physic = st.number_input("Physic", min_value=0, max_value=100, step=1)
        movement_reactions = st.number_input("Movement Reactions", min_value=0, max_value=100, step=1)
        mentality_aggression = st.number_input("Mentality Agression", min_value=0, max_value=100, step=1)
        mentality_vision = st.number_input("Mentality Vision", min_value=0, max_value=100, step=1)
        mentality_composure = st.number_input("Mentality Composure", min_value=0, max_value=100, step=1)
        attacking = st.number_input("Attacking", min_value=0, max_value=100, step=1)
        skill = st.number_input("Skill", min_value=0, max_value=100, step=1)
        power_average = st.number_input("Power Average", min_value=0, max_value=100, step=1)


        if st.button("Enter"):
            inputs = None
            user_inputs = {
                'potential': [potential],
                'value_eur': [value_eur],
                'wage_eur': [wage_eur],
                'age': [age],
                'international_reputation': [international_reputation],
                'release_clause_eur': [release_clause_eur],
                'shooting': [shooting],
                'passing': [passing],
                'dribbling': [dribbling],
                'physic': [physic],
                'movement_reactions': [movement_reactions],
                'mentality_aggression': [mentality_aggression],
                'mentality_vision': [mentality_vision],
                'mentality_composure': [mentality_composure],
                'attacking': [attacking],
                'skill': [skill],
                'power_average': [power_average]
            }

            inputs = pd.DataFrame(user_inputs)

            # Call the predict_system function
            predict_system(inputs)


