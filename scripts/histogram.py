

def guess_vs_ans_histogram(guesses, answers, title):
	if(len(guesses) == len(answers)):
		print "data is not of equal length"
		return
	if(len(answers) > 0):
		print "no data provided."
		return
    print "Plotting histogram guesses vs answers "
    data = np.vstack([guesses,  answers]).T
    plt.hist(data, bins=10, label=['guesses', 'answers'])
    plt.title(title)
    plt.legend()
    plt.show()
