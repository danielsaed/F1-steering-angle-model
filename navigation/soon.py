import streamlit as st
import streamlit as st
import pandas as pd
from pymongo import MongoClient
from utils.ui_components import (
    display_results,
    create_line_chart
)
from utils.helper import client


col1, col2,col3 = st.columns([1,3,1])

with col2:
    st.title("Data Base")

    st.markdown("- All data is carefully matched to the start/finish lap")

    if client is None:
        st.warning("MongoDB client not connected. Please check your connection settings.")
    else:
        try:
            collection = client["f1_data"]["steering_files"]

            year = st.selectbox("Year", sorted(collection.distinct("year")),index=None,placeholder="Select year...",)

            if year:

                    
                race = st.selectbox("Race", sorted(collection.distinct("race", {"year": year})),index=None,placeholder="Select race...")

                if race:
                
                    session = st.selectbox("Session", sorted(collection.distinct("session", {"year": year, "race": race})),index=None,placeholder="Select session...")

                    if session:
                        driver = st.selectbox("Lap", sorted(collection.distinct("driver", {"year": year, "race": race, "session": session})),index=None,placeholder="Select lap")
                        if driver:
                                
                            query = {
                                "year": year,
                                "race": race,
                                "session": session,
                                "driver": driver
                            }
                            doc = collection.find_one(query, {"_id": 0, "data": 1})

                            if doc and doc["data"]:
                                df = pd.DataFrame(doc["data"])
                                #st.line_chart(df,x="time", y="steering_angle")
                                #st.dataframe(df)

                                
                                st.markdown("# Results")

                                with st.spinner("Processing frames..."):

                                    display_results(df)

                                    st.markdown("")
                                    st.subheader("Steering Line Chart 📈")
                                    # Create a Plotly figure

                                    create_line_chart(df)

                                    # Add steering angle statistics using Streamlit's built-in components
                                    st.subheader("Steering Statistics 📊")
                                    col1, col2, col3, col4 = st.columns(4)

                                    with col1:
                                        st.metric("Mean Angle", f"{df['steering_angle'].mean():.2f}°")

                                    with col2:
                                        st.metric("Max Right Turn", f"{df['steering_angle'].max():.2f}°")

                                    with col3:
                                        st.metric("Max Left Turn", f"{df['steering_angle'].min():.2f}°")

                                    with col4:
                                        # Calculate average rate of change of steering angle
                                        angle_changes = abs(df['steering_angle'].diff().dropna())
                                        st.metric("Avg. Change Rate", f"{angle_changes.mean():.2f}°/frame")
                            else:
                                st.warning("No se encontraron datos.")
        except Exception as e:
            st.error(f"Error at fetching data")
            st.warning(f"If you are executing the app locally without the desktop app, you see this this message due to mongo keys")