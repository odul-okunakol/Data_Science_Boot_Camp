###############################################
# Python Exercises
###############################################

###############################################
# TASK 1: Examine the data types of data structures.
###############################################

x = 8
#integar

y = 3.2
#float

z = 8j + 18
#type(z)-->complex

a = "Hello World"
#string

b = True
#boolean

c = 23 < 22
#boolean


l = [1, 2, 3, 4,"String",3.2, False]
#liste


d = {"Name": "Jake",
     "Age": [27,56],
     "Adress": "Downtown"}
#dictionary

t = ("Machine Learning", "Data Science")
#tuble


s = {"Python", "Machine Learning", "Data Science","Python"}
#set



###############################################
# TASK 2: Convert all letters in the given string to uppercase.
# Replace commas and periods with a space, and split the sentence into words.
###############################################

text = "The goal is to turn data into information, and information into insight."

#answer2:
new_text=''
for i in text:
    new_text=text.upper()

new_text.replace(" ", " ,")



###############################################
# TASK 3: Perform the following tasks for the given list.
###############################################

lst = ["D","A","T","A","S","C","I","E","N","C","E"]

# Step 1: Check the number of elements in the given list.
len(lst)

# Step 2: Access the elements at index 0 and index 10.
lst[0]
lst[10]

# Step 3: Create a new list ["D", "A", "T", "A"] from the given list.
new_lst = [lst[0], lst[1], lst[2], lst[3]]
new_lst
# new_lst = lst[:4]

# Step 4: Remove the element at index 8.
lst.pop(8)
# lst.remove("N")
lst

# Step 5: Add a new element to the list.
lst.append("A")
lst


# Step 6: Add the element "N" back at index 8.
lst.insert(8, "N")
lst
###############################################
# TASK 4: Apply the following steps to the given dictionary structure.
###############################################

dict = {'Christian': ["America", 18],
        'Daisy': ["England", 12],
        'Antonio': ["Spain", 22],
        'Dante': ["Italy", 25]}

# Step 1: Access the keys.

dict.keys()

# Step 2: Access the values.

dict.values()

# Step 3: Update the value 12 associated with the key "Daisy" to 13.

dict.update({"Daisy": ["England", 13]})
dict

# Step 4: Add a new entry with key "Ahmet" and value [Turkey, 24].

dict.update({"Ahmet": ["Turkey", 24]})
dict

# Step 5: Remove "Antonio" from the dictionary.

dict.pop('Antonio')
dict

###############################################
# TASK 5: Write a function that takes a list as an argument,
# separates the even and odd numbers into different lists,
# and returns these lists.
###############################################

l = [2, 13, 18, 93, 22]

# Answer 5:
def separate_numbers(l):
    even_numbers = []
    odd_numbers = []
    for i in l:
        if i % 2 == 0:
            even_numbers.append(i)
        else:
            odd_numbers.append(i)
    return even_numbers, odd_numbers

even_numbers, odd_numbers = separate_numbers(l)

print("Even Numbers:", even_numbers)
print("Odd Numbers:", odd_numbers)
###############################################
# TASK 6: The list below contains the names of top-ranking students
# in engineering and medical faculties.
# The first three students represent the top ranks in the engineering faculty,
# while the last three students belong to the medical faculty.
# Use enumerate to print the rankings of students by faculty.
###############################################

students = ["Ali", "Veli", "Ayşe", "Talat", "Zeynep", "Ece"]

# Answer 6
for index, student in enumerate(students):
    if index < 3:
        print(f"Engineering Faculty - Student ranked {index + 1}: {student}")
    else:
        print(f"Medical Faculty - Student ranked {index - 2}: {student}")

###############################################
# TASK 7: Below are 3 lists containing course codes, credits, and quotas.
# Use zip to print out the course information.
###############################################

course_codes = ["CMP1005", "PSY1001", "HUK1005", "SEN2204"]
credits = [3, 4, 2, 4]
quotas = [30, 75, 150, 25]

# Answer 7
for code, credit, quota in zip(course_codes, credits, quotas):
    print(f"Course Code: {code}, Credit: {credit}, Quota: {quota}")
###############################################
# TASK 8: Two sets are given below.
# Define a function that checks:
# If the first set is a subset of the second set, print the difference (elements in set 2 but not in set 1).
# Otherwise, print the common elements between the two sets.
###############################################

set1 = set(["data", "python"])
set2 = set(["data", "function", "qcut", "lambda", "python", "miuul"])

common_elements = []
difference_elements = []

if set1.issubset(set2):
    difference_elements = set2.difference(set1)
else:
    common_elements = set1.intersection(set2)

difference_elements


