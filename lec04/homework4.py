def next_birthday(date, birthdays):
    """
    Find the next birthday after the given date.
    """
    # Extract and sort all birthday dates
    all_dates = sorted(birthdays.keys())
    
    # Loop through the sorted list
    for d in all_dates:
        # If this date is after the given date, return it
        if d > date:
            return d, birthdays[d]
    
    # If no later date found, wrap to the earliest one
    if all_dates:
        first_date = all_dates[0]
        return first_date, birthdays[first_date]
    else:
        # No birthdays in dictionary
        return (1, 1), []

    
