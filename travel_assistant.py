# travel_assistant.py
import os
import time
import json
import logging
import requests
import streamlit as st
from datetime import datetime, timedelta
from typing import Dict, List, Any

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ollama API configuration
OLLAMA_API_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2:latest"

# Initialize session state
if 'results' not in st.session_state:
    st.session_state.results = {}
if 'processing' not in st.session_state:
    st.session_state.processing = False

# Custom LLM function with retry logic
def query_llama(prompt: str, max_retries: int = 3, timeout: int = 120) -> str:
    """Query Ollama API with retry logic"""
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.3}
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.post(
                OLLAMA_API_URL,
                json=payload,
                timeout=timeout
            )
            response.raise_for_status()
            
            # Parse response
            response_data = response.json()
            if "response" in response_data:
                return response_data["response"]
            else:
                logger.error(f"Unexpected response format: {response_data}")
                return "Error: Invalid response format from LLM"
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Attempt {attempt+1} failed: {str(e)}")
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                logger.info(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                logger.error(f"All retries failed for prompt: {prompt[:50]}...")
                return f"Error: LLM request failed after {max_retries} attempts"
    
    return "Error: LLM request failed"

# Agent functions
def planner_agent(user_input: Dict[str, Any]) -> str:
    prompt = f"""
    You are a Senior Travel Planner. Create a {user_input['duration']}-day itinerary for a trip from {
        user_input['start_date']} to {user_input['end_date']}.
    Destination: {user_input['destination']}
    Travelers: {user_input['num_people']} {user_input['traveler_type']} travelers
    Special requests: {user_input['special_requests']}
    
    Output format:
    # Travel Itinerary for {user_input['destination']}
    ## Day 1: [Date]
    - [Time]: [Activity]
    - [Time]: [Activity]
    ...
    """
    return query_llama(prompt)

def experience_agent(user_input: Dict[str, Any]) -> str:
    prompt = f"""
    You are a Local Experience Specialist. Recommend activities in {user_input['destination']} for {
        user_input['traveler_type']} travelers.
    Special requests: {user_input['special_requests']}
    
    Output format:
    # Local Experiences in {user_input['destination']}
    ## Must-Try Activities
    - [Activity 1]: [Description]
    - [Activity 2]: [Description]
    
    ## Cultural Experiences
    - [Experience 1]: [Description]
    """
    return query_llama(prompt)

def recommendation_agent(user_input: Dict[str, Any]) -> str:
    prompt = f"""
    You are a Hospitality Concierge. Recommend hotels and transport in {user_input['destination']} for {
        user_input['num_people']} {user_input['traveler_type']} travelers.
    Budget level: {user_input['budget_level']}
    
    Output format:
    # Accommodations & Transport
    ## Hotels
    - [Option 1]: [Price range], [Features]
    
    ## Transportation
    - [Option 1]: [Description]
    """
    return query_llama(prompt)

def safety_agent(user_input: Dict[str, Any]) -> str:
    prompt = f"""
    You are a Travel Security Advisor. Provide safety tips for {user_input['destination']} during {
        user_input['start_date']} to {user_input['end_date']}.
    
    Output format:
    # Safety Information
    ## Travel Advisories
    - [Advisory 1]
    
    ## Health Recommendations
    - [Recommendation 1]
    
    ## Emergency Contacts
    - [Contact 1]
    """
    return query_llama(prompt)

def budget_agent(user_input: Dict[str, Any]) -> str:
    prompt = f"""
    You are a Travel Finance Manager. Estimate costs for {user_input['num_people']} people traveling to {
        user_input['destination']} for {user_input['duration']} days. Also suggest packing items.
    
    Output format:
    # Budget Estimate
    ## Cost Breakdown
    - Accommodation: [Estimate]
    - Transportation: [Estimate]
    - Total: [Total Estimate]
    
    # Packing List
    - [Category 1]
      - [Item 1]
      - [Item 2]
    """
    return query_llama(prompt)

# Streamlit UI Setup
st.set_page_config(
    page_title="AI Travel Assistant", 
    layout="wide", 
    page_icon="✈️",
    menu_items={
        'Get Help': 'https://github.com/your-repo',
        'About': "AI Travel Assistant v1.0"
    }
)

# Sidebar Inputs
with st.sidebar:
    st.header("✈️ Trip Details")
    today = datetime.today()
    
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=today)
    with col2:
        end_date = st.date_input("End Date", value=today + timedelta(days=7))
    
    from_location = st.text_input("Departing From", "New York")
    destination = st.text_input("Destination", "Paris, France")
    num_people = st.number_input("Number of Travelers", 1, 20, 2)
    traveler_type = st.selectbox("Traveler Type", [
        "Solo", "Couple", "Family with Kids", "Friends Group", 
        "Business", "Adventure Seekers", "Luxury Travelers"
    ])
    budget_level = st.selectbox("Budget Level", ["Budget", "Mid-range", "Luxury"])
    special_requests = st.text_area("Special Requirements", "Vegetarian food, accessible locations")
    
    if st.button("Plan My Trip", use_container_width=True, type="primary"):
        if start_date >= end_date:
            st.error("End date must be after start date")
        else:
            st.session_state.processing = True
            st.session_state.user_input = {
                "start_date": str(start_date),
                "end_date": str(end_date),
                "duration": (end_date - start_date).days,
                "from_location": from_location,
                "destination": destination,
                "num_people": num_people,
                "traveler_type": traveler_type,
                "budget_level": budget_level,
                "special_requests": special_requests
            }

# Main Content
st.title("AI Travel Assistant ✈️")
st.caption("Your personal travel planning expert powered by Ollama's LLaMA 3")

# Processing
if st.session_state.processing:
    with st.status("Planning your trip...", expanded=True) as status:
        user_input = st.session_state.user_input
        try:
            # Execute agents sequentially
            st.write("⏳ Creating itinerary...")
            st.session_state.results['itinerary'] = planner_agent(user_input)
            
            st.write("🎭 Finding experiences...")
            st.session_state.results['experiences'] = experience_agent(user_input)
            
            st.write("🏨 Selecting accommodations...")
            st.session_state.results['recommendations'] = recommendation_agent(user_input)
            
            st.write("⚠️ Checking safety...")
            st.session_state.results['safety'] = safety_agent(user_input)
            
            st.write("💰 Calculating budget...")
            budget_result = budget_agent(user_input)
            # Split budget and packing if possible
            if "Packing List" in budget_result:
                parts = budget_result.split("# Packing List")
                st.session_state.results['budget'] = parts[0]
                st.session_state.results['packing'] = "# Packing List" + parts[1] if len(parts) > 1 else ""
            else:
                st.session_state.results['budget'] = budget_result
                st.session_state.results['packing'] = "Packing list not generated"
            
            status.update(label="Trip plan ready! 🎉", state="complete")
            st.session_state.processing = False
            st.rerun()
            
        except Exception as e:
            st.error(f"Planning failed: {str(e)}")
            st.session_state.processing = False
            status.update(label="Planning failed", state="error")
            logger.exception("Planning failed")

# Display Results
if st.session_state.results:
    tabs = st.tabs([
        "📅 Itinerary", "🎭 Experiences", "🏨 Accommodations", 
        "⚠️ Safety", "💰 Budget & Packing"
    ])
    
    with tabs[0]:
        st.subheader("Daily Itinerary")
        if 'itinerary' in st.session_state.results:
            st.markdown(st.session_state.results['itinerary'])
        else:
            st.warning("Itinerary not generated")
    
    with tabs[1]:
        st.subheader("Activities & Experiences")
        if 'experiences' in st.session_state.results:
            st.markdown(st.session_state.results['experiences'])
        else:
            st.warning("Experiences not generated")
    
    with tabs[2]:
        st.subheader("Hotels & Transport")
        if 'recommendations' in st.session_state.results:
            st.markdown(st.session_state.results['recommendations'])
        else:
            st.warning("Recommendations not generated")
    
    with tabs[3]:
        st.subheader("Safety Information")
        if 'safety' in st.session_state.results:
            st.markdown(st.session_state.results['safety'])
        else:
            st.warning("Safety information not generated")
    
    with tabs[4]:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Budget Estimate")
            if 'budget' in st.session_state.results:
                st.markdown(st.session_state.results['budget'])
            else:
                st.warning("Budget not generated")
        with col2:
            st.subheader("Packing List")
            if 'packing' in st.session_state.results:
                st.markdown(st.session_state.results['packing'])
            else:
                st.warning("Packing list not generated")
    
    st.download_button(
        "Download Full Plan", 
        json.dumps(st.session_state.results, indent=2), 
        file_name="travel_plan.json",
        mime="application/json"
    )
    
    if st.button("Create New Plan", type="primary", use_container_width=True):
        st.session_state.results = {}
        st.rerun()
        
else:
    st.info("Enter your travel details in the sidebar to get started")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.write("### How it works:")
        st.markdown("""
        1. Enter your trip details
        2. Click 'Plan My Trip'
        3. Get personalized recommendations:
           - Daily itinerary
           - Local experiences
           - Hotel & transport options
           - Safety information
           - Budget estimate
        """)
    with col2:
        st.image("https://images.unsplash.com/photo-1503220317375-aaad61436b1b", 
                 caption="Your perfect trip starts here")

# Footer
st.divider()
st.caption("© 2024 AI Travel Assistant | Powered by Ollama LLaMA 3")