from flask import Flask, render_template
from sklearn.linear_model import LogisticRegression
from datetime import datetime, timedelta
import pickle

# import the logistic regresion object to fit new cases
logreg = pickle.load(open('logreg.logisticRegression', 'rb'))

# counter to constrain the number of pings to the API
cntr = 0
max_pings = 10

# for constraining the number of pings based on time intervals
dt = datetime.now()

# url input string parser
def url_string_parse(input_string, feature_split='&', key_value_split='='):
	"""Parse the url input and create a dictionary. Every feature supplied via the url is split by the feature_split
	and the feature name and feature value is split by the key_value_split.
	The function returns a dictionary.
	"""
	input_vec = input_string.split(feature_split)
	return {el.split(key_value_split)[0]: el.split(key_value_split)[1] for el in input_vec}

# creating a flask instance
app = Flask(__name__)

# api to be called at this url
@app.route('/titanic/<string:input_vec>', methods=['GET'])
def live_or_die(input_vec):
	""" It parses the string supplied by the url to get the three features to be passed into the
	logistic regression object. The features are 'class', 'sex' and 'age'.
	It also imposes restrictions on the number of times the the api can be pinged per minute.
	The function returns a json object or a string objectwith the predicted outcome.
	"""
	global cntr, max_pings, dt

	# if one minute has passed, reset everything
	if (dt <= (datetime.now() - timedelta(minutes=1))): dt = datetime.now(); cntr = 0

	# every time the function is called, i.e. for every api call, increment the counter by one
	cntr += 1

	# Impose restrictions on the api call
	if (cntr > max_pings):
		return render_template('api.html', limit_reached = (cntr > max_pings),max_pings=max_pings,
								pings=str(cntr), timedate=str(dt.replace(second=0, microsecond=0)),
								wait_time=str(60 - (datetime.now() - dt).seconds))

	# parse features and values from url string to dictionary
	v = url_string_parse(input_vec)

	v['sex'] = 0 if v['sex'] == 'male' else 1
	v['class'] = int(v['class'])
	v['age'] = float(v['age'])

	# predict the outcome
	v['Outcome'] = 'Dies' if logreg.predict([[v['class'], v['sex'], v['age']]]) == 0 else 'Survives'

	prefx = 'She ' if v['sex'] == 1 else 'He '

	return render_template('api.html', prefix=prefx, outcome=v['Outcome'], limit_reached = (cntr > max_pings),
							pings=str(cntr), timedate=str(dt.replace(second=0, microsecond=0)))

@app.route('/titanic/', methods=['GET'])
def guide():
	return render_template('index.html')

@app.route('/titanic/doc/', methods=['GET'])
def my_model():
	""" Render the model methodology for reproducability
	"""
	return render_template('DS_api.html')

# Run the flask application
if __name__ == '__main__':
	app.run(debug=True)
