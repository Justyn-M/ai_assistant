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
            # This doesnt actually work for current implementations due to an unknown Google issue
            # Set reminders manually universally on Google calender. Can do multiple reminders.
        ]

    event = {
        'summary': summary,
        'start': {'dateTime': event_datetime, 'timeZone': 'Australia/Perth'},
        'end': {'dateTime': event_datetime, 'timeZone': 'Australia/Perth'},
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
    """Updates an event's time, date, or name in Google Calendar."""
    service = authenticate_calendar()
    now = datetime.datetime.utcnow().isoformat() + 'Z'

    events_result = service.events().list(
        calendarId='justyn2s04mahen23@gmail.com', timeMin=now,
        singleEvents=True, orderBy='startTime'
    ).execute()

    events = events_result.get('items', [])

    for event in events:
        if event['summary'] == summary:
            if new_summary:
                event['summary'] = new_summary

            if new_date or new_time:
                original_datetime = event['start'].get('dateTime')
                if original_datetime:
                    dt = datetime.datetime.fromisoformat(original_datetime[:-1])  # Remove 'Z'
                    updated_date = new_date or dt.strftime("%Y-%m-%d")
                    updated_time = new_time or dt.strftime("%H:%M")
                    updated_datetime = f"{updated_date}T{updated_time}:00"
                    event['start']['dateTime'] = updated_datetime
                    event['end']['dateTime'] = updated_datetime

            updated_event = service.events().update(
                calendarId='justyn2s04mahen23@gmail.com',
                eventId=event['id'], body=event
            ).execute()

            return f"Event '{summary}' updated successfully."

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
        'start': {'dateTime': event_datetime, 'timeZone': 'Australia/Perth'},
        'end': {'dateTime': event_datetime, 'timeZone': 'Australia/Perth'},
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
    """Checks the user's busy slots and suggests free periods."""
    service = authenticate_calendar()

    now = datetime.datetime.utcnow()
    end_time = now + datetime.timedelta(days=1)

    now_iso = now.isoformat() + 'Z'
    end_iso = end_time.isoformat() + 'Z'

    events_result = service.events().list(
        calendarId='justyn2s04mahen23@gmail.com',
        timeMin=now_iso,
        timeMax=end_iso,
        singleEvents=True,
        orderBy='startTime'
    ).execute()

    events = events_result.get('items', [])

    if not events:
        response = "You're completely free for the next 24 hours~ Spend all that time with me, okay? ♡\n"

    # Sort and parse event times
    busy_times = []
    for event in events:
        start_raw = event['start'].get('dateTime')
        end_raw = event['end'].get('dateTime')
        summary = event.get('summary', 'Unnamed Event')

        if start_raw and end_raw:
            start = datetime.datetime.fromisoformat(start_raw[:-1])  # Remove 'Z'
            end = datetime.datetime.fromisoformat(end_raw[:-1])
            busy_times.append((start, end, summary))

    # Sort events
    busy_times.sort(key=lambda x: x[0])

    # Build free time blocks
    free_blocks = []
    current = now
    for start, end, summary in busy_times:
        if start > current:
            free_blocks.append((current, start))
        current = max(current, end)

    if current < end_time:
        free_blocks.append((current, end_time))

    # Format response
    response = "You're busy during:\n"
    for start, end, summary in busy_times:
        response += f"- {summary}: {start.strftime('%H:%M')} to {end.strftime('%H:%M')}\n"

    response += "\nAnd you're free:\n"
    for start, end in free_blocks:
        response += f"- From {start.strftime('%H:%M')} to {end.strftime('%H:%M')}\n"

    return response
