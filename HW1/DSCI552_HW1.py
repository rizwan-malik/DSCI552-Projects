### Team Members #####
# Muhammad Rizwan Malik
# Hamza Belal Kazi

# DSCI 552  -  HW1

import csv
import numpy as np
import math


# This is the data structure to store the tree. It will store:
# Feature = The attribute on the current node.
# Value = The type of attribute on which it is being split. Basically the branch from the current node.
# Children = A list which stores the children on the current node. Right now I am a little iffy about this
# dictionary data structure, because i think the way we are implementing it, we might not need a dictionary at all.
# It can be stored simply as a tuple. I will discuss it with you over zoom.
class Node:
	def __init__(self, feature):
		self.feature = feature
		self.children = []

	# This method adds children to the parent. Children themselves will be tuples (node, feature_value or branch) and
	# they will be appended to the parent's list named children.
	def add_child(self, node, branch):
		self.children.append((node, branch))


def get_subtables(table, column):
	subtables = []
	unique = np.unique(table[:, column])
	for i in unique:
		indices = np.argwhere(table[:, column] == i)
		subtable = np.array([])
		for j in indices:
			subtable = np.append(subtable, [table[j, :]])
		subtable = np.reshape(subtable, (len(indices), 7))
		subtables.append(subtable)
	# subtables = np.reshape(subtables, (len(unique), 7))
	# print(len(subtables))
	return subtables


def get_entropy(table):
	unique, count = np.unique(table[:, -1], return_counts=True)
	if len(count) == 1:
		entropy = 0
	else:
		total = count[0] + count[1]
		p = count[1] / total
		entropy = p * math.log2(1 / p) + (1 - p) * math.log2(1 / (1 - p))
	# print(unique, count)
	# print(entropy)
	return entropy


# This method is for printing the tree. It will print the tree at different indented levels. It was a very simple
# trick but it took me an unnecessarily long time to figure out.
def print_method(attr, level, spaces):
	temp_attr = "Node Depth " + str(level) + " : " + str(Attr_names[attr.feature])
	attr_spacing = len(temp_attr) + spaces + 3
	print(temp_attr.rjust(attr_spacing))
	# print("Attribute at Tree Depth " + str(level) + " : " + str(Attr_names[attr.feature]))
	for child in attr.children:
		temp_case = "Case : " + str(child[1])
		case_spacing = len(temp_case) + spaces + 5
		print(" ")  ## for newline between seperate cases
		print(temp_case.rjust(case_spacing))

		if len(child[0].children) > 0:
			print_method(child[0], level + 1, spaces + 5)
		else:
			temp_leaf = "(Leaf Node) Enjoyed : " + str(child[0].feature)
			leaf_spacing = len(temp_leaf) + spaces + 10
			print(temp_leaf.rjust(leaf_spacing))


def get_InfoGain(table, attribute):
	EntropyBeforeSplit = get_entropy(table)
	subs = get_subtables(table, attribute)
	ave_Entropy = 0
	for loop in range(0, len(subs)):
		ave_Entropy = ave_Entropy + get_entropy(subs[loop]) * (len(subs[loop]) / len(table))
	InfoGain = EntropyBeforeSplit - ave_Entropy
	return InfoGain


# This method will be called recursively and in the end it will return the root of the tree.
def decision_tree(table, features):
	# These are the two end-cases I could think of, i think there will be another one as well. But these can be added
	# easily.
	# If there's only single type of option left in last column.
	if len(np.unique(table[:, -1])) == 1:
		# This simply returns the node data structure of the leaf node which will simply be a Yes or a No.
		return Node(np.unique(table[:, -1])[0])  # lea value passed stored back
	# if there are no features/attributes left to split upon. We will simply select the result based on majority of
	# Yes or No.
	if len(features) == 0:
		unique, count = np.unique(table[:, -1], return_counts=True)
		Leaf_value = unique[np.argmax(count)]
		return Node(Leaf_value)

	# This is where the gain will be calculated and split will happen and after that it will create the Node (current
	# node), remove the current attribute from feature list and then recall the decision_tree function. After that
	# call, it will add the result (whatever decision tree returns) to its child list. Then it will re-insert the
	# attribute, in the end it will return the current node (which will basically be the root node, when all the
	# recursive calls end). Last two statements need to be implemented which now seem pretty straightforward and I
	# think we'll be able to do a big portion of the assignment tomorrow.
	IG = np.zeros(len(features))
	i = 0
	for attr in features:
		IG[i] = get_InfoGain(table, attr)
		i = i + 1
	index = np.argmax(IG)
	temp_node = features[index]
	current_node = Node(temp_node)
	subs = get_subtables(table, temp_node)
	features.remove(temp_node)

	for st in subs:
		branch = st[0, temp_node]
		childNode = decision_tree(st, features)  # look at if Flist copy need to be re filled with deleted
		current_node.add_child(childNode, branch)  # sp[0,temp node] give value on outgoing branch to reach to this
	# child node
	features.append(temp_node)
	features.sort()

	return current_node


# After the above statement we will remove the current attribute and recall the same method with the new feature
# list. ALso, I just realised that we will need to enclose the statements after line 115 in a for loop, which will
# loop over the unique types of the current attribute, therefore, the table that we will pass in the recursive calls
# will be the modified table for that specific type of attribute.


def parseData():
	data = open('dt_data.txt', 'r')
	data.readline()  # ignore first read in line
	data.readline()  # ignore empty second line
	table2D = []  # np.empty(shape=(1,22))
	i = 0
	spl_char = ': '
	for line in data:
		fullrow = line  # data.readline()
		row = fullrow.partition(spl_char)[2]  # return 3rd part of tuple: post splitter
		row = row[0:len(row) - 2]
		array = row.split(',')
		table2D.append(array)  # np.append(table2D,np.array(array) )
		i = i + 1

	table2D = np.asarray(table2D)
	return table2D


# We can write this method to make an automatic predition for whichever question we enter in the form of the question
# given below and it will simply make comparisons and keep going until it hits a leaf node.
def makePrediction(question, tree):
	if tree.feature == ' Yes' or tree.feature == ' No':  # base case
		return tree.feature

	Attr_names = ["Occupied", "Price", "Music", "Location", "VIP", "Favorite beer"]
	temp_node = tree.feature
	value = question[Attr_names[temp_node]]

	for child in tree.children:

		if child[1] == value:
			return makePrediction(question, child[0])


# here call deci tree with 0,....6 array of features
table2D = parseData()
FeatureList = [0, 1, 2, 3, 4, 5]
root = decision_tree(table2D, FeatureList)

Attr_names = ["Occupied", "Price", "Music", "Location", "VIP", "Favorite beer"]
print_method(root, 0, 0)
question = {'Occupied': 'Moderate', 'Price': ' Cheap', 'Music': ' Loud', 'Location': ' City-Center', 'VIP': ' No',
            'Favorite beer': ' No'}

print("Answer to requested prediction  :   ")
print(makePrediction(question, root))
