from google.oauth2 import service_account
from googleapiclient.discovery import build
import datetime

SCOPES = ['https://www.googleapis.com/auth/calendar']

def authenticate_calendar():
    """Authenticates and returns a Google Calendar API service object."""
    creds = service_account.Credentials.from_service_account_file(
        'credentials.json', scopes=SCOPES
    )
    service = build('calendar', 'v3', credentials=creds)
    return service


def add_event(summary, date, time, reminders=None, color_id=1):
    """Adds an event with multiple reminders and color customization to Google Calendar."""
    service = authenticate_calendar()
    
    event_datetime = f"{date}T{time}:00"
    
    # Default reminders if none provided
    if reminders is None:
        reminders = [
            {'method': 'popup', 'minutes': 30},  # Pop-up 30 minutes before
            {'method': 'email', 'minutes': 60}   # Email 1 hour before
        ]

    event = {
        'summary': summary,
        'start': {'dateTime': event_datetime, 'timeZone': 'UTC'},
        'end': {'dateTime': event_datetime, 'timeZone': 'UTC'},
        'colorId': str(color_id),  # Convert color_id to string
        'reminders': {
            'useDefault': False,
            'overrides': reminders
        }
    }

    event = service.events().insert(calendarId='justyn2s04mahen23@gmail.com', body=event).execute()
    return f"Event '{summary}' added with color {color_id} and custom reminders!"


def get_upcoming_events():
    """Fetches upcoming events from Google Calendar."""
    service = authenticate_calendar()
    
    now = datetime.datetime.utcnow().isoformat() + 'Z'  # 'Z' indicates UTC time
    events_result = service.events().list(
        calendarId='primary', timeMin=now, maxResults=10, singleEvents=True,
        orderBy='startTime').execute()
    
    events = events_result.get('items', [])
    
    if not events:
        return "No upcoming events found."

    event_list = []
    for event in events:
        start = event['start'].get('dateTime', event['start'].get('date'))
        event_list.append(f"{start} - {event['summary']}")

    return "\n".join(event_list)


def delete_event(summary, date=None):
    """Deletes an event from Google Calendar based on event name and optional date."""
    service = authenticate_calendar()

    now = datetime.datetime.utcnow().isoformat() + 'Z'
    
    # Fetch events matching the given summary
    events_result = service.events().list(
        calendarId='justyn2s04mahen23@gmail.com', timeMin=now,
        singleEvents=True, orderBy='startTime'
    ).execute()
    
    events = events_result.get('items', [])
    
    for event in events:
        event_summary = event['summary']
        event_start = event['start'].get('dateTime', event['start'].get('date'))

        # If the event name matches and (if given) the date matches
        if event_summary == summary and (date is None or event_start.startswith(date)):
            service.events().delete(calendarId='justyn2s04mahen23@gmail.com', eventId=event['id']).execute()
            return f"Event '{summary}' on {event_start} deleted from Google Calendar!"

    return f"No matching event found for '{summary}' on {date}."

def update_event(summary, new_date=None, new_time=None, new_summary=None):
    """Updates an event's details in Google Calendar."""
    service = authenticate_calendar()

    now = datetime.datetime.utcnow().isoformat() + 'Z'
    
    events_result = service.events().list(
        calendarId='justyn2s04mahen23@gmail.com', timeMin=now,
        singleEvents=True, orderBy='startTime'
    ).execute()

    events = events_result.get('items', [])
    
    for event in events:
        if event['summary'] == summary:
            # Update the event details
            if new_summary:
                event['summary'] = new_summary
            if new_date and new_time:
                event['start']['dateTime'] = f"{new_date}T{new_time}:00"
                event['end']['dateTime'] = f"{new_date}T{new_time}:00"

            updated_event = service.events().update(
                calendarId='justyn2s04mahen23@gmail.com', eventId=event['id'], body=event
            ).execute()

            return f"Event '{summary}' updated successfully!"

    return f"No event found with name '{summary}'."


def get_events_by_date(date):
    """Fetches events for a specific date."""
    service = authenticate_calendar()

    start_date = f"{date}T00:00:00Z"
    end_date = f"{date}T23:59:59Z"

    events_result = service.events().list(
        calendarId='justyn2s04mahen23@gmail.com', timeMin=start_date,
        timeMax=end_date, singleEvents=True, orderBy='startTime'
    ).execute()

    events = events_result.get('items', [])

    if not events:
        return f"No events found for {date}."

    return "\n".join([f"{event['start'].get('dateTime', event['start'].get('date'))} - {event['summary']}" for event in events])

def add_recurring_event(summary, start_date, start_time, recurrence_rule):
    """Adds a recurring event to Google Calendar."""
    service = authenticate_calendar()
    
    event_datetime = f"{start_date}T{start_time}:00"
    
    event = {
        'summary': summary,
        'start': {'dateTime': event_datetime, 'timeZone': 'UTC'},
        'end': {'dateTime': event_datetime, 'timeZone': 'UTC'},
        'recurrence': [recurrence_rule]
    }

    event = service.events().insert(calendarId='justyn2s04mahen23@gmail.com', body=event).execute()
    return f"Recurring event '{summary}' added with rule: {recurrence_rule}"

def delete_past_events():
    """Deletes all past events from Google Calendar."""
    service = authenticate_calendar()

    now = datetime.datetime.utcnow().isoformat() + 'Z'
    
    events_result = service.events().list(
        calendarId='justyn2s04mahen23@gmail.com', timeMax=now,
        singleEvents=True, orderBy='startTime'
    ).execute()

    events = events_result.get('items', [])
    
    if not events:
        return "No past events to delete."

    for event in events:
        service.events().delete(calendarId='justyn2s04mahen23@gmail.com', eventId=event['id']).execute()
    
    return f"{len(events)} past events deleted!"


def check_free_time():
    """Checks when the user is available."""
    service = authenticate_calendar()

    now = datetime.datetime.utcnow().isoformat() + 'Z'
    end_time = (datetime.datetime.utcnow() + datetime.timedelta(days=7)).isoformat() + 'Z'

    events_result = service.events().list(
        calendarId='justyn2s04mahen23@gmail.com', timeMin=now, timeMax=end_time,
        singleEvents=True, orderBy='startTime'
    ).execute()

    events = events_result.get('items', [])

    if not events:
        return "Your calendar is free!"

    busy_slots = [event['start'].get('dateTime', event['start'].get('date')) for event in events]
    return f"You're busy on: \n" + "\n".join(busy_slots)


def test_google_calendar_functions():
    print("\n--- Testing Google Calendar Functions ---\n")

    # 1️⃣ Add a single event
    print("✅ Testing: Adding an event")
    print(add_event("Meeting with Client", "2025-03-15", "10:00"))

    # 2️⃣ Get upcoming events
    print("\n✅ Testing: Fetching upcoming events")
    print(get_upcoming_events())

    # 3️⃣ Get events by specific date
    print("\n✅ Testing: Fetching events on a specific date")
    print(get_events_by_date("2025-03-15"))

    # 4️⃣ Update an existing event
    print("\n✅ Testing: Updating an event")
    print(update_event("Meeting with Client", new_date="2025-03-16", new_time="11:00", new_summary="Client Meeting Updated"))

    # 5️⃣ Delete an event
    print("\n✅ Testing: Deleting an event")
    print(delete_event("Client Meeting Updated", "2025-03-16"))

    # 6️⃣ Add an event with custom reminders & color
    print("\n✅ Testing: Adding an event with multiple reminders and custom color")
    print(add_event("Dentist Appointment", "2025-03-18", "09:00", 
                    reminders=[
                        {'method': 'popup', 'minutes': 15},  # 15 min pop-up
                        {'method': 'email', 'minutes': 120}  # 2 hours email
                    ],
                    color_id=6))  # Light Green

    # 7️⃣ Add a recurring event (every Monday)
    print("\n✅ Testing: Adding a recurring event")
    print(add_recurring_event("Weekly Team Meeting", "2025-03-17", "14:00", 
                              "RRULE:FREQ=WEEKLY;BYDAY=MO"))

    # 8️⃣ Check available time slots
    print("\n✅ Testing: Checking free time in the next 7 days")
    print(check_free_time())

    # 9️⃣ Delete past events
    print("\n✅ Testing: Deleting past events")
    print(delete_past_events())

    print("\n--- Test Completed Successfully! ---")


if __name__ == "__main__":
    # 6️⃣ Add an event with custom reminders & color
    print("\n✅ Testing: Adding an event with multiple reminders and custom color")
    print(add_event("Dentist Appointment", "2025-03-18", "09:00", 
                    reminders=[
                        {'method': 'popup', 'minutes': 15},  # 15 min pop-up
                        {'method': 'email', 'minutes': 120}  # 2 hours email
                    ],
                    color_id=6))