from flask import Flask,jsonify,request
from flask_cors import CORS
from ChatGPT_Paraphraser import paraphrase
from SummarizationModel import summarize
app = Flask(__name__)
CORS(app)

@app.route("/")
def hello_world():
    ans=paraphrase("This is the best application ever made")
    return jsonify({"answer":ans})

@app.route('/data', methods=['GET'])
def get_data():
    data = request.json  # Assuming the request body contains JSON data

    if data and 'name' in data and 'prompt' in data:
        name = data['name']
        prompt = data['prompt']
        # Process the data (you can add more logic here)
        ans=paraphrase(prompt)
        return jsonify({"answer":ans})
    else:
        return jsonify({"error": "Please provide 'name' and 'prompt' in the request body as JSON."})
@app.route('/paraphraser', methods=['GET'])
def get_prompt():
    prompt=request.args.get("prompt")
    print(prompt)
    ans=paraphrase(prompt)
    return jsonify({"answer":ans})
@app.route('/summary', methods=['GET'])
def get_summary():
    prompt=request.args.get("prompt")
    print(prompt)
    ans=summarize(prompt)
    return jsonify({"answer":ans})
if __name__ == '__main__':
  
    app.run(debug = True,port=1234)

# We will run using python3 main.py

# Use this command to run the server
# flask --app hello run //Will run in port 5000