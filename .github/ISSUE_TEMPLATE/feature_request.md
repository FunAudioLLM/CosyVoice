def feature_request():
    # Collect inputs from the user
    title = input("Enter the title of your feature request: ")
    labels = input("Enter any labels for this request (comma separated): ")
    assignees = input("Enter the assignees for this request (comma separated): ")

    problem_description = input("Is your feature request related to a problem? Please describe: ")
    solution_description = input("Describe the solution you'd like: ")
    alternatives = input("Describe alternatives you've considered: ")
    additional_context = input("Any additional context or screenshots you'd like to provide? ")

    # Format the feature request as per the template
    feature_request_template = f"""
    ---
    name: Feature request
    about: Suggest an idea for this project
    title: {title}
    labels: {labels}
    assignees: {assignees}
    ---

    *Is your feature request related to a problem? Please describe.*
    {problem_description}

    *Describe the solution you'd like*
    {solution_description}

    *Describe alternatives you've considered*
    {alternatives}

    *Additional context*
    {additional_context}
    """
    
    print("Your feature request has been created as follows:\n")
    print(feature_request_template)

# Run the feature request function
feature_request()
