from flask import Flask, jsonify, render_template
import pandas as pd
import os
import subprocess
import sys

app = Flask(
    __name__,
    static_url_path="/outputs",
    static_folder="outputs"
)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/status")
def status():
    return jsonify({
        "project": "AirPure Hospital Grid",
        "status": "running",
        "message": "Hospital ventilation monitoring system is online"
    })


@app.route("/api/results")
def results():

    file_path = "outputs/control_actions.csv"

    if not os.path.exists(file_path):
        return jsonify({
            "error": "AI results file not found"
        }), 404

    df = pd.read_csv(file_path)

    return jsonify(df.to_dict(orient="records"))


@app.route("/api/run", methods=["POST"])
def run_analysis():

    try:

        result = subprocess.run(
            [sys.executable, "main.py"],
            capture_output=True,
            text=True,
            timeout=300
        )

        if result.returncode != 0:

            return jsonify({
                "success": False,
                "error": result.stderr,
                "output": result.stdout
            }), 500

        return jsonify({
            "success": True,
            "message": "AI analysis completed successfully",
            "output": result.stdout
        })

    except subprocess.TimeoutExpired:

        return jsonify({
            "success": False,
            "error": "AI analysis timed out."
        }), 500

    except Exception as e:

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


if __name__ == "__main__":
    app.run(debug=True)
