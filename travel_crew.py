import os
import time
import json
import logging
import requests
import streamlit as st
from datetime import datetime, timedelta
from typing import Dict, List, Any
from crewai import Agent, Task, Crew, Process, LLM

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize session state
if 'results' not in st.session_state:
    st.session_state.results = {}
if 'processing' not in st.session_state:
    st.session_state.processing = False

# Ollama configuration
OLLAMA_MODEL = "ollama/llama3.2:latest"
OLLAMA_BASE_URL = "http://localhost:11434"
# Ensure the shared_llm is correctly initialized as a CrewAI LLM instance
# The LLM class in CrewAI does not take base_url directly in its constructor for Ollama.
# It expects the model name. CrewAI handles the Ollama connection via environment variables or direct config if needed.
# For simplicity and common use, ensure Ollama is running and accessible.
# If you need to explicitly set base_url for CrewAI's LLM with Ollama, you might need a custom LLM integration or
# rely on CrewAI's internal Ollama handling (often through model mapping or env vars).
# However, for a basic setup with "ollama/llama3.2:latest", this generally works if Ollama is running.
shared_llm = LLM(model=OLLAMA_MODEL) # Simplified for typical CrewAI Ollama integration

# Function to get available Ollama models
def get_ollama_models() -> List[str]:
    """Fetch available models from Ollama API"""
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        response.raise_for_status()
        models_data = response.json()
        return [f"ollama/{model['name']}" for model in models_data.get('models', [])]
    except Exception as e:
        logger.error(f"Failed to fetch models: {str(e)}")
        # Fallback to default if fetching fails, ensuring at least one model is available
        return [OLLAMA_MODEL]

# Agent definitions
def create_agents(model: str):
    """Create CrewAI agents with the specified model"""
    # Create a new LLM instance with the selected model
    # It's better to pass the model string directly to the LLM constructor for CrewAI
    # shared_llm already uses OLLAMA_MODEL, if selected_model is different, you'd re-initialize
    # For now, let's assume selected_model is indeed OLLAMA_MODEL from sidebar
    agent_llm = LLM(model=model)

    planner = Agent(
        role='Senior Travel Planner',
        goal='Create personalized travel itineraries',
        backstory="Expert travel planner with 20+ years experience crafting perfect schedules",
        llm=agent_llm,
        verbose=True
    )

    experience = Agent(
        role='Local Experience Specialist',
        goal='Recommend unique local experiences and cultural activities',
        backstory="Cultural anthropologist and world traveler with deep local knowledge",
        llm=agent_llm,
        verbose=True
    )

    recommendation = Agent(
        role='Hospitality Concierge',
        goal='Suggest accommodations, transportation, and dining options',
        backstory="Former five-star hotel manager with premium service connections",
        llm=agent_llm,
        verbose=True
    )

    safety = Agent(
        role='Travel Security Advisor',
        goal='Provide safety information and travel advisories',
        backstory="Ex-government security specialist with 15 years in travel risk management",
        llm=agent_llm,
        verbose=True
    )

    budget = Agent(
        role='Travel Finance Manager',
        goal='Create budget estimates and packing suggestions',
        backstory="Financial planner specializing in travel economics",
        llm=agent_llm,
        verbose=True
    )

    return planner, experience, recommendation, safety, budget

# Create tasks
def create_tasks(user_input: Dict[str, Any], agents: tuple):
    """Create CrewAI tasks based on user input"""
    planner, experience, recommendation, safety, budget = agents

    itinerary_task = Task(
        description=f"""Create a detailed {user_input['duration']}-day itinerary for a trip from {
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
        Ensure the output is pure markdown, with no extra text or explanations.
        """,
        agent=planner,
        expected_output="Markdown formatted daily itinerary"
    )

    experience_task = Task(
        description=f"""Recommend unique and local activities in {user_input['destination']} for {
            user_input['traveler_type']} travelers. Focus on cultural, adventure, and local experiences.
        Special requests: {user_input['special_requests']}

        Output format:
        # Local Experiences in {user_input['destination']}
        ## Must-Try Activities
        - [Activity 1]: [Description]
        - [Activity 2]: [Description]

        ## Cultural Experiences
        - [Experience 1]: [Description]
        - [Experience 2]: [Description]
        Ensure the output is pure markdown, with no extra text or explanations.
        """,
        agent=experience,
        expected_output="Curated list of experiences with descriptions in markdown"
    )

    recommendation_task = Task(
        description=f"""Recommend hotels, transportation, and dining options in {user_input['destination']} for {
            user_input['num_people']} {user_input['traveler_type']} travelers.
        Budget level: {user_input['budget_level']}

        Output format:
        # Accommodations & Transport
        ## Hotels
        - [Hotel Option 1]: [Price range], [Features], [Location]
        - [Hotel Option 2]: [Price range], [Features], [Location]

        ## Transportation
        - [Transport Option 1]: [Description]
        - [Transport Option 2]: [Description]

        ## Dining Suggestions
        - [Restaurant Name 1]: [Cuisine], [Price range], [Notes based on special requests]
        - [Restaurant Name 2]: [Cuisine], [Price range], [Notes based on special requests]
        Ensure the output is pure markdown, with no extra text or explanations.
        """,
        agent=recommendation,
        expected_output="Structured recommendations with options for hotels, transport, and dining in markdown"
    )

    safety_task = Task(
        description=f"""Provide comprehensive safety tips and travel advisories for {user_input['destination']} for the period {
            user_input['start_date']} to {user_input['end_date']}. Include health recommendations and essential emergency contacts.

        Output format:
        # Safety Information for {user_input['destination']}
        ## Travel Advisories
        - [Advisory 1]: [Details]
        - [Advisory 2]: [Details]

        ## Health Recommendations
        - [Recommendation 1]: [Details]
        - [Recommendation 2]: [Details]

        ## Emergency Contacts
        - [Contact Name/Service 1]: [Number]
        - [Contact Name/Service 2]: [Number]
        Ensure the output is pure markdown, with no extra text or explanations.
        """,
        agent=safety,
        expected_output="Safety briefing document with categorized information in markdown"
    )

    budget_task = Task(
        description=f"""Estimate costs for {user_input['num_people']} people traveling to {
            user_input['destination']} for {user_input['duration']} days, considering a {user_input['budget_level']} budget.
            Also, suggest a comprehensive packing list based on the destination and travel dates.

        Output format:
        # Budget Estimate for {user_input['destination']} Trip
        ## Cost Breakdown (Estimates)
        - Accommodation: [Estimate per person/total, e.g., $X/night or $Y total]
        - Transportation: [Estimate per person/total]
        - Food & Dining: [Estimate per person/total]
        - Activities/Experiences: [Estimate per person/total]
        - Miscellaneous: [Estimate per person/total]
        - **Total Estimated Cost:** [Total Estimate]

        # Packing List for {user_input['destination']} Trip
        ## Clothing
        - [Item 1]
        - [Item 2]
        ## Essentials
        - [Item 1]
        - [Item 2]
        ## Health & Safety
        - [Item 1]
        - [Item 2]
        Ensure the output is pure markdown, with no extra text or explanations.
        """,
        agent=budget,
        expected_output="Budget breakdown and categorized packing list in markdown"
    )

    return itinerary_task, experience_task, recommendation_task, safety_task, budget_task


# Streamlit UI Setup
st.set_page_config(
    page_title="AI Travel Assistant",
    layout="wide",
    page_icon="✈️",
    menu_items={
        'Get Help': 'https://github.com/your-repo', # Replace with your actual repo if applicable
        'About': "AI Travel Assistant v1.0"
    }
)

# Sidebar - Model Selection
with st.sidebar:
    st.header("🧠 AI Model Settings")

    # Get available models
    available_models = get_ollama_models()
    # Ensure OLLAMA_MODEL is always in available_models for consistent initial selection
    if OLLAMA_MODEL not in available_models:
        available_models.insert(0, OLLAMA_MODEL) # Add it to the top if not found

    selected_model = st.selectbox(
        "Choose AI Model",
        available_models,
        index=available_models.index(OLLAMA_MODEL) if OLLAMA_MODEL in available_models else 0
    )
    st.caption(f"Selected: {selected_model}")

    if st.button("Refresh Models"):
        # This will re-run the script, causing get_ollama_models to be called again
        st.rerun()

    st.divider()

# Sidebar - Trip Details
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
            # Reset results when a new plan is initiated
            st.session_state.results = {}
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
            # Rerun immediately to show "Planning your trip..." status
            st.rerun()


# Main Content
st.title("AI Travel Assistant ✈️")
st.caption(f"Your personal travel planning expert powered by Ollama's {selected_model}")

# Check if Ollama is running
def check_ollama_health():
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3) # Shorter timeout
        return response.status_code == 200
    except requests.exceptions.RequestException as e:
        logger.error(f"Ollama health check failed: {e}")
        return False

# Show warning if Ollama not running
if not check_ollama_health():
    st.error(f"Ollama service not detected at {OLLAMA_BASE_URL}. Please ensure Ollama is running.")
    st.info("You can download and run Ollama from [ollama.com](https://ollama.com). After installing, make sure to pull the `llama3.2` model if you haven't already by running `ollama run llama3.2` in your terminal.")
    if st.button("Try Again", type="secondary"):
        st.rerun()
    st.stop() # Stop execution if Ollama is not running

# Processing logic (only runs if st.session_state.processing is True)
if st.session_state.processing:
    with st.status("Planning your trip...", expanded=True) as status:
        user_input = st.session_state.user_input
        try:
            # Create agents and tasks using the selected model
            agents = create_agents(selected_model)
            itinerary_task, experience_task, recommendation_task, safety_task, budget_task = create_tasks(user_input, agents)

            # Form the crew
            travel_crew = Crew(
                agents=agents,
                tasks=[itinerary_task, experience_task, recommendation_task, safety_task, budget_task],
                process=Process.sequential,
                verbose=True, # Keep verbose for debugging in console
                memory=False
            )

            st.write(f"🚀 Starting travel planning with {selected_model}...")
            # Execute the crew
            crew_result = travel_crew.kickoff() # crew_result typically holds the final output of the last task

            # --- FIX: Extracting raw output from tasks ---
            # Accessing .raw attribute for the actual string output
            results = {
                'itinerary': itinerary_task.output.raw if itinerary_task.output else "No itinerary generated.",
                'experiences': experience_task.output.raw if experience_task.output else "No experiences generated.",
                'recommendations': recommendation_task.output.raw if recommendation_task.output else "No recommendations generated.",
                'safety': safety_task.output.raw if safety_task.output else "No safety info generated.",
                'budget': budget_task.output.raw if budget_task.output else "No budget generated.",
            }

            # --- FIX: Handle packing list extraction from the raw budget string ---
            # Now, results['budget'] is already the raw string.
            budget_output_raw = results['budget']
            packing_list_marker = "# Packing List"

            if packing_list_marker in budget_output_raw:
                parts = budget_output_raw.split(packing_list_marker, 1) # Split only once
                results['budget'] = parts[0].strip() # Budget part
                results['packing'] = packing_list_marker + parts[1].strip() # Packing list part
            else:
                # If the marker isn't found, assume the budget part is the whole string
                results['packing'] = "Packing list section not found in budget output."

            st.session_state.results = results
            status.update(label="Trip plan ready! 🎉", state="complete")
            st.session_state.processing = False
            st.rerun() # Rerun to display the results in the main section

        except Exception as e:
            st.error(f"Planning failed: {str(e)}")
            st.session_state.processing = False
            status.update(label="Planning failed", state="error", expanded=True)
            logger.exception("CrewAI planning failed")
            # Clear results if planning failed, or show an error state
            st.session_state.results = {"error": f"Failed to generate plan: {e}"}
            st.rerun() # Rerun to reflect the error state or clear results

# Display Results (only if st.session_state.results is populated and not empty)
if st.session_state.results and not st.session_state.processing:
    # Check if an error occurred in results
    if "error" in st.session_state.results:
        st.error(st.session_state.results["error"])
        # Provide an option to retry or start a new plan after an error
        if st.button("Try New Plan", type="primary", use_container_width=True):
            st.session_state.results = {}
            st.rerun()
    else:
        tabs = st.tabs([
            "📅 Itinerary", "🎭 Experiences", "🏨 Accommodations",
            "⚠️ Safety", "💰 Budget & Packing"
        ])

        with tabs[0]:
            st.subheader("Daily Itinerary")
            st.markdown(st.session_state.results.get('itinerary', "No itinerary available."))

        with tabs[1]:
            st.subheader("Activities & Experiences")
            st.markdown(st.session_state.results.get('experiences', "No experiences available."))

        with tabs[2]:
            st.subheader("Hotels & Transport")
            st.markdown(st.session_state.results.get('recommendations', "No recommendations available."))

        with tabs[3]:
            st.subheader("Safety Information")
            st.markdown(st.session_state.results.get('safety', "No safety information available."))

        with tabs[4]:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Budget Estimate")
                st.markdown(st.session_state.results.get('budget', "No budget estimate available."))
            with col2:
                st.subheader("Packing List")
                st.markdown(st.session_state.results.get('packing', "Packing list not available."))

        # --- FIX STARTS HERE ---
        # Retrieve user_input from session_state as it's persistent
        current_user_input = st.session_state.get('user_input', {}) # Use .get with a default empty dict for safety

        # Convert results to a display-friendly JSON string for download
        downloadable_results = {k: v for k, v in st.session_state.results.items() if isinstance(v, str)}

        # Use current_user_input to construct file names
        destination_name = current_user_input.get('destination', 'plan').replace(' ', '_')
        start_date_str = current_user_input.get('start_date', datetime.now().strftime('%Y-%m-%d'))


        st.download_button(
            "Download Full Plan (Markdown)",
            "\n\n---\n\n".join(downloadable_results.values()), # Join all markdown sections
            file_name=f"travel_plan_{destination_name}_{start_date_str}.md",
            mime="text/markdown"
        )
        st.download_button(
            "Download Full Plan (JSON)",
            json.dumps(st.session_state.results, indent=2), # Keep JSON for full data structure
            file_name=f"travel_plan_{destination_name}_{start_date_str}.json",
            mime="application/json"
        )
        # --- FIX ENDS HERE ---

        if st.button("Create New Plan", type="primary", use_container_width=True):
            st.session_state.results = {}
            # Clear user_input when starting a new plan, if desired
            if 'user_input' in st.session_state:
                del st.session_state.user_input
            st.rerun()

else:
    # Initial state or after a new plan is requested
    st.info("Enter your travel details in the sidebar to get started!")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.write("### How it works:")
        st.markdown("""
        1. Select an AI model
        2. Enter your trip details (destination, dates, travelers, etc.)
        3. Click 'Plan My Trip'
        4. Get personalized recommendations across multiple categories:
           - 📅 **Daily itinerary**
           - 🎭 **Local experiences**
           - 🏨 **Hotel & transport options**
           - ⚠️ **Safety information**
           - 💰 **Budget estimate & packing list**
        """)
        st.write("### Available Models:")
        if available_models:
            st.write(", ".join([m.replace("ollama/", "") for m in available_models]))
        else:
            st.write("No Ollama models found. Please ensure Ollama is running and models are downloaded.")

    with col2:
        st.image("https://images.unsplash.com/photo-1503220317375-aaad61436b1b",
                 caption="Your perfect trip starts here")

# Footer
st.divider()
st.caption("© 2024 AI Travel Assistant | Powered by CrewAI and Ollama")