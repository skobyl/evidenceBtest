from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("result_exercise_events.csv")
print(df.columns.tolist())

unique_students = list(sorted(list(set(df['user_id'].tolist()))))

# print(unique_students)

unique_exos = list(sorted(list(set(df["category_data_learning_item_id"].tolist()))))
# print("EXOS: ", unique_exos[:100])
print("nbr unique exos: ", len(unique_exos))

# replace user ids
student_id_mapping = {old_id: new_id for new_id, old_id in enumerate(unique_students, start=1)}

df['user_id'] = df['user_id'].map(student_id_mapping)

# replace exo ids
exo_id_mapping = {old_id: new_id for new_id, old_id in enumerate(unique_exos, start=1)}

df['category_data_learning_item_id'] = df['category_data_learning_item_id'].map(exo_id_mapping)

# average score for each student
average_scores = df.groupby('user_id')['data_score'].mean()

# count nbr of exercises for each student
exercise_counts = df.groupby('user_id')['category_data_learning_item_id'].count()
user_per_exercise_counts = df.groupby('category_data_learning_item_id')['user_id'].count()
print(user_per_exercise_counts)

# combine results into one df: user id, average score, nbr exos
summary_df = pd.DataFrame({
    'average_score': average_scores,
    'exercise_count': exercise_counts
}).reset_index()

print(summary_df.head())

#**************************************************************
# CALCULATE ELO PER STUDENT AND PER EXO: progress + final scores
#**************************************************************
def track_elo_progress(df, k=32, initial_elo=1000):

    student_elo = defaultdict() # {student id: final elo}
    exercise_elo = defaultdict() # {exo id: final elo}
    progress_all = [] # [{event nbr: nbr, student id: id, exo: id, time: time dur, elo score: current score}]
    student_elo_progress = defaultdict(list) # {student id: [elo 1, elo 2, elo 3...]}

    for index, row in df.iterrows():
        student_id = row['user_id']
        exercise_id = row['category_data_learning_item_id']
        actual_score = row['data_score']
        time = row["data_duration"]

        # Initialize Elo scores
        if student_id not in student_elo:
            student_elo[student_id] = initial_elo
        if exercise_id not in exercise_elo:
            exercise_elo[exercise_id] = initial_elo

        # Current ratings
        student_rating = student_elo[student_id]
        exercise_rating = exercise_elo[exercise_id]

        # Calculate expected performance
        expected_score = 1 / (1 + 10 ** ((exercise_rating - student_rating) / 400))

        # Update ratings
        student_elo[student_id] += k * (actual_score - expected_score)
        exercise_elo[exercise_id] += k * (expected_score - actual_score)

        # Log student progress
        progress_all.append({
            'interaction': index + 1,
            'student_id': student_id,
            'exercise_id': exercise_id,
            'actual_score': actual_score,
            'expected_score': expected_score,
            'student_elo': student_elo[student_id],
            'exercise_elo': exercise_elo[exercise_id],
            'time': time
        })

        student_elo_progress[student_id].append(student_elo[student_id])

    return progress_all, student_elo, exercise_elo, student_elo_progress



# calculate student elo score progress
progress_elo, student_elo, exercise_elo, student_elo_progress = track_elo_progress(df)
progress_df = pd.DataFrame(progress_elo)
print(progress_df.columns.tolist())
# add nbr events/interactions
progress_df['student_interaction'] = progress_df.groupby('student_id').cumcount() + 1


# get time range: mix max
min_time_per_exercise = progress_df.groupby("exercise_id")["time"].min()
max_time_per_exercise = progress_df.groupby("exercise_id")["time"].max()


# get average time spent and average elo score per exercise
avg_data = progress_df.groupby('exercise_id').agg({'time': 'mean', 'exercise_elo': 'mean'}).reset_index()

# convert time to log, because of a large difference between min and max time spent on the exo
avg_data["time_log"] = np.log(avg_data["time"])

# get a subset of data, only exos with elo > 1000
avg_data_subset = avg_data[ (avg_data["exercise_elo"] >= 1000) ]

#**********************************************
# VISUALIZE DIFFICULT EXOS (scatterplot)
# x axis - time, y axis - elo score
#**********************************************

plt.figure(figsize=(10, 6))
plt.scatter(avg_data_subset['time'], avg_data_subset['exercise_elo'], alpha=0.7)

# add labels for each exercise on the plot
for i, row in avg_data_subset.iterrows():
    plt.text(row['time'], row['exercise_elo'], str(row['exercise_id']), fontsize=10, ha='right')

plt.title('Average time spent on exo vs average ELO score per exo', fontsize=14)
plt.xlabel('Average time Log', fontsize=12)
plt.ylabel('Average ELO score', fontsize=12)

plt.grid(True, linestyle='--', alpha=0.5)
plt.show()


# rename df columns
df = df.rename(columns={"category_data_learning_item_id": "exercise_id"})
df = df.rename(columns={"user_id": "student_id"})

# add the obtained elo scores to the initial df
df['exercise_elo'] = df['exercise_id'].map(exercise_elo)
df["student_elo"] = df["student_id"].map(student_elo)
# print(df)


#******************************************************
# SELECT THE MOST DIFFICULT EXO AND STUDENTS WHO DID IT
#******************************************************

# select the most difficult exo
df_subset = df[df["exercise_id"] == 810]

# select a subset of students who did the exercise

# selected_students = df_subset.user_id.tolist()

# selected_students = [705, 1045, 1327, 507, 49, 1271, 2898, 702, 2875, 1972,
#                      2252, 2523, 1716, 2331, 2592, 2421]

selected_students = [1327, 507, 49, 702, 2875, 1972,
                     2252, 2523, 1716]

student_elo_subset = {key:value for key, value in student_elo.items() if key in selected_students}
sorted_dict = dict(sorted(student_elo_subset.items(), key=lambda item: item[1]))

print(sorted_dict)
print(df_subset.columns.tolist())

elo_progression_subset = {key:value for key, value in student_elo_progress.items() if key in selected_students}

max_events = 0
initial_elo = 1000

# calculate the max number of events for the selected students: will be used in x-axis in the progression plot
for student_id in selected_students:
    student_elo_progress[student_id].insert(0, initial_elo)
    if len(student_elo_progress[student_id]) > max_events:
        max_events = len(student_elo_progress[student_id])
max_events += 1

#********************************************************
# VISUALIZE THE SELECTED STUDENTS' PROGRESS FOR THE DIFFICULT EXO
#*********************************************************

plt.figure(figsize=(14, 8))
# sort students by their final elo score
sorted_students = sorted(student_elo_subset.items(), key=lambda x: x[1], reverse=True)

# plot the progression of the selected students one by one
for student_id,_ in sorted_students:
    plt.plot(
        range(1, len(student_elo_progress[student_id]) + 1),
        student_elo_progress[student_id],
        marker="o",
        label=f"Student {student_id} (Final ELO: {student_elo_subset[student_id]:.2f})"
    )


plt.title("ELO Score Progression for Selected Students", fontsize=16)
plt.xlabel("Event Number", fontsize=12)
plt.ylabel("ELO Score", fontsize=12)
plt.legend(loc="upper left", bbox_to_anchor=(1.05, 1), fontsize=10)
plt.grid(alpha=0.5)
plt.xticks(range(0, max_events, 1))
plt.tight_layout()
plt.show()

#******************************************************************
# VISUALIZE THE ACTUAL SCORES FOR EXO 810 FOR THE SELECTED STUDENTS
#******************************************************************

# categorize ELO scores into groups: low, high, medium

def categorize_elo(elo):
    if elo < 1000:
        return 'Low ELO'
    elif 1000 <= elo < 1100:
        return 'Medium ELO'
    else:
        return 'High ELO'

selected_exercise = 810

filtered_data = df[(df["exercise_id"] == selected_exercise) & (df["student_id"].isin(selected_students))]

filtered_data['data_elo'] = filtered_data['student_id'].map(student_elo_subset)

filtered_data['elo_category'] = filtered_data['data_elo'].apply(categorize_elo)

# assign colors to elo categories for visualization
colors = {'Low ELO': 'red', 'Medium ELO': 'blue', 'High ELO': 'green'}
elo_category_descriptions = {
    'Low ELO': 'Low ELO (< 1000)',
    'Medium ELO': 'Medium ELO (1000-1100)',
    'High ELO': 'High ELO (> 1100)'
}

# plot actual scores vs time spent, grouped by elo categories
plt.figure(figsize=(10, 6))

for _, row in filtered_data.iterrows():
    plt.scatter(
        row['data_duration'],
        row['data_score'],
        color=colors[row['elo_category']],
        label=elo_category_descriptions[row['elo_category']],
        s=100,
        alpha=0.7
    )


# Remove duplicate legend entries
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=10)

# Add labels, title, and grid
plt.title('Scores vs Time Spent by Students on Exercise 810 (Grouped by ELO)', fontsize=16)
plt.xlabel('Time Spent (seconds)', fontsize=12)
plt.ylabel('Actual Score', fontsize=12)
plt.grid(alpha=0.5)
plt.tight_layout()
plt.show()