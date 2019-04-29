import pandas as pd
import numpy as np
import random


def get_check_ins_by_location(location_file, check_ins_file):
	location = pd.read_csv(location_file, sep=',', header=0)
	# print(us_ca_ids[0:10])
	check_ins = pd.read_csv(check_ins_file, sep='\t', header=None)
	us_ca_check_ins = check_ins[check_ins[1].isin(location['id'])]
	# a = pd.merge(right=us_ca, left=us_ca_check_ins, right_on='id', left_on=1, how='inner')
	# print(a)
	# print(len(check_ins))
	# print(len(us_ca_check_ins))
	return us_ca_check_ins


def get_user_sequence(check_ins, min_sequence_length):
	# x1 = check_ins.groupby(0).filter(lambda x: len(x) >= min_sequence_length).groupby(0)["index"]
	# x1 = check_ins.groupby(0).filter(lambda x: len(x) >= min_sequence_length).groupby(0)[1]
	x1 = check_ins.groupby(0)[1]
	x1_list = x1.apply(list)
	return x1_list.values


def convert_to_lda_input(user_sequences, min_sequence_length, min_num_place, location_file, split_ratio, folder_path, filename_prefix):
	location = pd.read_csv(location_file, sep=',', header=0)
	location_ids = location["id"].values
	tests = []
	trains = []
	validations = []
	random_places = []
	for sequence in user_sequences:
		place_set = list(set(sequence))
		if len(place_set) >= min_num_place and len(sequence) >= min_sequence_length:
			place_indexes = []
			train = []
			validation = []
			for place in place_set:
				place_indexes.append(np.where(location_ids == place)[0][0])

			place_index = place_indexes[place_set.index(sequence[-1])]
			tests.append(place_index)

			random_false = random.randint(0, len(location_ids))
			while random_false in place_indexes:
				random_false = random.randint(0, len(location_ids))
			random_places.append(random_false)

			sequence = sequence[:-1]
			random.shuffle(sequence)
			for place in sequence:
				place_index = place_indexes[place_set.index(place)]
				if random.uniform(0.01, 1.0) <= split_ratio and len(train) < len(sequence) * 0.8:
					train.append(place_index)
				else:
					validation.append(place_index)

			trains.append(train)
			validations.append(validation)
	# write_file(trains, folder_path + filename_prefix + "_" + str(min_sequence_length) + "_train.dat")
	# write_file(validations, folder_path + filename_prefix + "_" + str(min_sequence_length) + "_validation.dat")
	# with open(folder_path + filename_prefix + "_" + str(min_sequence_length) + "_test.dat", "w") as outputFile:
	# 	outputFile.write("\n".join(map(str, tests)))
	with open(folder_path + filename_prefix + "_" + str(min_sequence_length) + "_random.dat", "w") as outputFile:
		outputFile.write("\n".join(map(str, random_places)))


def write_file(data, output_filename):
	with open(output_filename, "w") as outputFile:
		for sequence in data:
			sequence_set = set(sequence)
			outputFile.write(str(len(sequence_set)) + " ")
			for place in sequence_set:
				token = str(place) + ":" + str(sequence.count(place))
				outputFile.write(token + " ")
			outputFile.write("\n")


loc_file = "D:/Research/Dataset/checkin/us_canada.txt"
checkins_file = "D:/Research/Dataset/checkin/Gowalla_totalCheckins_chekin10.txt"
us_check_ins = get_check_ins_by_location(loc_file, checkins_file)
sequences = get_user_sequence(us_check_ins, 5)
convert_to_lda_input(sequences, 6, 2, loc_file, 0.8, "D:/Research/Dataset/checkin/New folder/", "us")
