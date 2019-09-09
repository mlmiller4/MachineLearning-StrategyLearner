import numpy as np
import random as rd
import pandas as pd


class RTLearner(object):

    def __init__(self, leaf_size=1, verbose=False):
        # pass # move along, these aren't the drones you're looking for
        self.tree = None
        self.leaf_size = leaf_size

    def author(self):
        return 'mmiller319' # replace tb34 with your Georgia Tech username

    def addEvidence(self, dataX, dataY):
        """
        @summary: Add training data to learner
        @param dataX: X values of data to add
        @param dataY: the Y training values
        """

        # Format dataY as an ndarray
        dataY = np.array([dataY])

        # Transpose dataY so it can be appended onto dataX
        dataY_transpose = dataY.T
        data_appended = np.append(dataX, dataY_transpose, axis=1)

        # Pass appended data to the build_tree function
        self.tree = self.build_tree(data_appended)

    # Implements the Random Tree Algorithm of A Cutler
    # Basically the same as Quinlan's Decision Tree Alg but using the mean of two randomly chosen
    # rows in the data to determine the split value
    def build_tree(self, data):

        # Base Case - when the number of nodes is < leaf_size, the stopping condition has been reached.
        if data.shape[0] < self.leaf_size:
            return np.array([['Leaf', np.mean(data[:, -1]), None, None]])

        # Check if all values of data.y are the same - if so return a leaf
        # .unique checks how many unique values are in data - if its size =1, then all values are the same
        if np.unique(data[:, -1].size == 1):
            return np.array([['Leaf', data[0, -1], None, None]])

        else:

            # Instead of finding the best feature to split based on correlations, the Random Tree
            # Alg. simply chooses a feature at random

            randomFeature = rd.randint(0, data.shape[1]-2)
            #randomFeature = data[:, randNum]
            SplitVal = np.median(data[:, randomFeature])

            # If the split value is equal to the maximum value for the best feature, then no right
            # subtree will be formed.
            if SplitVal == max(data[:, randomFeature]):
                return np.array([['Leaf', np.mean(data[:, -1]), None, None]])

            # Otherwise, create a left and right subtree bases on whether the feature values are
            # greater than or less than the split value
            leftSubTree = self.build_tree(data[data[:, randomFeature] <= SplitVal])
            rightSubTree = self.build_tree(data[data[:, randomFeature] > SplitVal])

            # Create an np.array as the root, append left and right subtrees to it and return the full tree
            root = np.array([[randomFeature, SplitVal, 1, leftSubTree.shape[0] + 1]])
            halfTree = np.append(root, leftSubTree, axis=0)
            fullTree = np.append(halfTree, rightSubTree, axis=0)
            return fullTree


    def query(self, points):

        results = []    # Hold results


        # for each row, predict the value of the target (EM)
        for i in range(0, points.shape[0]):

            currentRow = points[i,:]    # get current row from points array

            treeRow = 0     # current row in walking of the tree

            while (self.tree[treeRow, 0] != 'Leaf'):
                currTreeRow = int(self.tree[treeRow, 0])
                SplitVal = self.tree[treeRow, 1]

                if currentRow[currTreeRow] <= SplitVal:
                    treeRow += int(self.tree[treeRow, 2])
                else:
                    treeRow += int(self.tree[treeRow, 3])

            # Get feature found in tree based on walking tree using the points[] array and append to results[]
            currentFeature = self.tree[treeRow, 1]
            results.append(currentFeature)

        return results


if __name__ == "__main__":
    print ""
