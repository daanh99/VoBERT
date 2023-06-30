def unseen_heuristic():
    # This function is used to evaluate the performance of the heuristic method
    # on unseen data. The results are written to a csv file.
    # The heuristic method just classifies all unseen logkeys as anomalies.
    # Unseen logkeys are logkeys that occur in a sequence in the test set, but in no sequence in the training set.



    # write results to csv file
    with open(self.output_dir + "results/" + name + ".csv", "a") as f:
        f.write("{},{},{},{},{},{},{},{},{},{},{},{}\n".format(self.ratio_unseen[1:], thr, TP, TN, FP, FN, P, R, F1,
                                                               MCC, auc, samples))
    # write results to csv file


