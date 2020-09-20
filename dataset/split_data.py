import sys

categorie = "sadness" 
f=open( "all_data" + categorie + "/" + categorie + ".txt", "r", encoding='utf8', errors='ignore')
lcat = len(categorie)
if f.mode == 'r':
	lines = f.readlines()
	for i in range(0,len(lines)):
		line = lines[i]
		line = line[6:]
		k = 0
		while(line[k:k+lcat] != categorie or k==len(line)-1):
			k = k + 1
		if( line[k:k+lcat] == categorie ) :
			if(line[k+lcat+1]>= '0' and  line[k+lcat+1]<= '9'):
				intensity = float(line[k+lcat+1:k+lcat+6])
				if(intensity >= 0.25):
					print(line[:k])
					namefile = "splited_data" + categorie + "/sample_" + str(i) + ".txt"
					e = open(namefile,"w+", encoding='utf8', errors='ignore')
					e.write(line[:k])
					e.close
		
		
		
			