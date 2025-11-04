def next_birthday(date, birthdays):
    """
    Find the next birthday after the given date.

    @param:
    date - a tuple of two integers specifying (month, day)
    birthdays - a dict mapping from date tuples to lists of names, for example,
      birthdays[(1,10)] = list of all people with birthdays on January 10.

    @return:
    birthday - the next day, after given date, on which somebody has a birthday
    list_of_names - list of all people with birthdays on that date
    """
    if not birthdays:
        return (1, 1), []

    # Sort all birthday dates
    all_dates = sorted(birthdays.keys())

    # Try to find a birthday after the given date
    for d in all_dates:
        if d[0] > date[0] or (d[0] == date[0] and d[1] > date[1]):
            return d, birthdays[d]

    # If not found, wrap around to the first birthday in the next year
    first = all_dates[0]
    return first, birthdays[first]

    
