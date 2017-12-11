from flask import Flask, jsonify
from sklearn.linear_model import LogisticRegression
import pickle

logreg = pickle.load(open('logreg.logisticRegression', 'rb'))
app = Flask(__name__)

@app.route('/titanic/<string:input_vec>', methods=['GET'])
def live_or_die(input_vec):
	input_vec = input_vec.split('&')
	v = {el.split('=')[0]: el.split('=')[1] for el in input_vec}

	v['sex'] = 0 if v['sex'] == 'male' else 1
	v['class'] = int(v['class'])
	v['age'] = float(v['age'])
	v['Outcome'] = 'Dies' if logreg.predict([[v['class'], v['sex'], v['age']]]) == 0 else 'Survives'

	return jsonify(v)

# @app.route('/titanic/', methods=['GET'])
# def guide():
# 	return """
# 	Structure of GET request is:<br>
# 	http://127.0.0.1:5000/titanic/class=number 1 or 2 or 3&sex=male or female&age=some float number<br><br>
# 	Examples:<br>http://127.0.0.1:5000/titanic/class=3&sex=male&age=20.5<br>
# 	http://127.0.0.1:5000/titanic/class=1&sex=female&age=45<br>
# 	"""

if __name__ == '__main__':
	app.run(debug=True)