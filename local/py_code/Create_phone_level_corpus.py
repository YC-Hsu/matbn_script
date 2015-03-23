# -*- coding: utf-8 -*-
import sys,re,math

def Copy_list(list) :
	re_list=[]
	for i in list :
		re_list.append(i)
	return re_list

if len(sys.argv) != 3 :
	print "Usage: [python-command] " + sys.argv[0] + " [options] <phone-feats> <phone-corpus>"
	print "e.g.: python " + sys.argv[0] + " exp/MisPronunciation.txt \\"
	print "\texp/MisPronunciation_test.txt"
	print "options: "
	print "			"
else :
	#讀取<phone-feats>檔案
	ifp = open(sys.argv[1], 'r')
	ofp = open(sys.argv[2], 'w')
	out_flag = 0
	total_read = 0
	total_write = 0
	while True :
		corpus=[]
		index=0
		if out_flag == 1 :
                        break
		while True :
			if index >= 10000 :
				break
			line = ifp.readline()
			if not line :
				out_flag = 1
				break
			line_list = re.split(' |\n',line)
			line_list.remove('')
			line_list.insert(1,'T')
			corpus.append(line_list)
			index += 1
		print "Read " + str(len(corpus)) + " feats.(True)"
		total_read += len(corpus)

		rand_num = [15863,88656,32455,73596,18723,85172,47446,25789,841264,46646,17896,85742,74111,956464,8318,7569,8674,831582]
		T_corpus_num = len(corpus)
		for i in range(0,T_corpus_num) :
			for j in rand_num :
				if corpus[i][0] != corpus[(i+j)%T_corpus_num][0] :
					corpus.append(Copy_list(corpus[(i+j)%T_corpus_num]))
					corpus[len(corpus)-1][1] = 'F'
					corpus[len(corpus)-1][0] = corpus[i][0]
					break
		print "write " + str(len(corpus)) + " feats.(True+Flase)"
		total_write += len(corpus)
		corpus.sort()

		print "Saving file " + sys.argv[2] + " ... "
		for i in range(0,len(corpus)) :
			ofp.write(corpus[i][0])
			for j in range(1,len(corpus[i])) :
				ofp.write(" " + corpus[i][j])
			ofp.write("\n")
	ifp.close()
	ofp.close()
	print "Total Read " + str(total_read) + " feats."
	print "Total Write " + str(total_write) + " feats."
	print "Done."

