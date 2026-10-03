import numpy as np
import pandas as pd
import joblib
import smtplib
import random 
from datetime import datetime

from flask import Flask,render_template,request,redirect,session,jsonify,flash
from flask_mysqldb import MySQL
from tensorflow.keras.models import load_model
from apscheduler.schedulers.background import BackgroundScheduler

app = Flask(__name__)
app.secret_key="maintainx_secret"
# -------------------------
# SYSTEM MODE
# -------------------------
SYSTEM_MODE = "AUTO"   # AUTO / MANUAL
# -------------------------
# DATABASE CONFIG
# -------------------------

app.config["MYSQL_HOST"]="localhost"
app.config["MYSQL_USER"]="root"
app.config["MYSQL_PASSWORD"]="navya@123456789"
app.config["MYSQL_DB"]="maintainx_ai"

mysql = MySQL(app)

def get_cursor():
    return mysql.connection.cursor()

# -------------------------
# LOAD MODELS
# -------------------------

try:
    xgb_model = joblib.load("model/models/xgboost_model.pkl")
    scaler = joblib.load("model/models/scaler.pkl")
    lstm_model = load_model("model/models/lstm_model.h5")
    print("Models loaded successfully")
except Exception as e:
    print("Model loading error:",e)

# -------------------------
# AUTH HELPERS
# -------------------------

def login_required():
    if "user_id" not in session:
        return redirect("/")

def admin_required():
    if "user_id" not in session:
        return redirect("/")
    if session.get("role")!="admin":
        return redirect("/dashboard")

# -------------------------
# RISK + RECOMMENDATION
# -------------------------
def risk_info(prob):

    if prob < 0.05:
        return "Low Risk","Machine operating normally"

    elif prob < 0.15:
        return "Medium Risk","Inspect vibration and cooling system"

    else:
        return "High Risk","Service required within 5 days"
#--------------------------
# simulated data generator
#--------------------------
import random
from datetime import datetime

# store machine state (important for realistic pattern)
machine_state = {}

def generate_simulated_data():
     if SYSTEM_MODE != "AUTO":
       return
     with app.app_context():

        cur = get_cursor()

        cur.execute("SELECT machineID FROM machines")
        machines = cur.fetchall()

        for m in machines:
            machineID = m[0]

            # -------------------------
            # INIT MACHINE STATE
            # -------------------------
            if machineID not in machine_state:
                machine_state[machineID] = {
                    "health": 100,
                    "mode": "normal"
                }

            state = machine_state[machineID]

            # -------------------------
            # MODE TRANSITIONS
            # -------------------------
            if state["mode"] == "normal" and random.random() < 0.1:
                state["mode"] = "degrading"

            elif state["mode"] == "degrading" and random.random() < 0.1:
                state["mode"] = "failure"

            elif state["mode"] == "failure" and random.random() < 0.3:
                state["mode"] = "maintenance"

            elif state["mode"] == "maintenance":
                state["mode"] = "normal"
                state["health"] = 100

            # -------------------------
            # GENERATE DATA BASED ON MODE
            # -------------------------

            if state["mode"] == "normal":
                volt = 120 + random.uniform(-3,3)
                rotate = 300 + random.uniform(-10,10)
                pressure = 60 + random.uniform(-3,3)
                vibration = 20 + random.uniform(-1,1)

            elif state["mode"] == "degrading":
                state["health"] -= random.uniform(0.5, 2)

                volt = 120 + random.uniform(0,10)
                rotate = 300 + random.uniform(10,30)
                pressure = 60 + random.uniform(5,15)
                vibration = 20 + random.uniform(5,15)

            elif state["mode"] == "failure":
                state["health"] -= random.uniform(5,10)

                volt = 150 + random.uniform(10,20)
                rotate = 400 + random.uniform(30,60)
                pressure = 90 + random.uniform(20,40)
                vibration = 50 + random.uniform(20,40)

            elif state["mode"] == "maintenance":
                volt = 115 + random.uniform(-2,2)
                rotate = 290 + random.uniform(-5,5)
                pressure = 55 + random.uniform(-2,2)
                vibration = 18 + random.uniform(-1,1)

            # -------------------------
            # INSERT TELEMETRY
            # -------------------------
            cur.execute("""
            INSERT INTO telemetry(machineID, datetime, volt, rotate, pressure, vibration)
            VALUES(%s,%s,%s,%s,%s,%s)
            """,(machineID, datetime.now(), volt, rotate, pressure, vibration))

            # -------------------------
            # KEEP LAST 50 RECORDS
            # -------------------------
            cur.execute("""
            DELETE FROM telemetry
            WHERE machineID = %s AND datetime NOT IN (
                SELECT datetime FROM (
                    SELECT datetime FROM telemetry
                    WHERE machineID = %s
                    ORDER BY datetime DESC
                    LIMIT 50
                ) AS temp
            )
            """,(machineID, machineID))

        mysql.connection.commit()

        print("✅ Smart simulated telemetry generated")


# -------------------------
# MODE SWITCH
# -------------------------
@app.route("/set_mode/<mode>")
def set_mode(mode):

    global SYSTEM_MODE

    if mode in ["AUTO", "MANUAL"]:
        SYSTEM_MODE = mode

    return redirect("/dashboard")
# -------------------------
# EMAIL ALERT
# -------------------------

def send_alert(machine, risk, current_prob, future_prob, recommendation, health):

    try:
        sender = "automonitoring.alerts@gmail.com"
        password = "jlzgxrvxfdfyamzr"
        receiver = "automonitoring.admin@gmail.com"

        
        time_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        msg = f"""Subject:  MaintainX Alert - {risk}

Machine ID: {machine}

Risk Level: {risk}
Current Failure Probability: {round(current_prob*100,2)}%
Future Failure Probability: {round(future_prob*100,2)}%

Health Score: {round(health,2)}

Recommendation:
{recommendation}

Timestamp: {time_now}

Action Required:
Please inspect the machine immediately if risk is HIGH.

- MaintainX Auto Monitoring System
"""

        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(sender, password)
        server.sendmail(sender, receiver, msg)
        server.quit()

        print(" Alert sent successfully")

    except Exception as e:
        print("Email failed:", e)


# -------------------------
# PREDICTION LOGIC
# -------------------------
def predict_machine(machineID, data):

    

    # latest row (FIXED)
    latest = data.iloc[-1].values

    columns = ["volt","rotate","pressure","vibration"]
    latest_df = pd.DataFrame(latest.reshape(1,-1), columns=columns)

    latest_scaled = scaler.transform(latest_df)

    # XGBoost prediction
    current_prob = xgb_model.predict_proba(latest_scaled)[0][1]

    health = (1 - current_prob) * 100

    # -------------------------------
    # LSTM sequence preparation (FIXED)
    # -------------------------------
    seq = []

    for row in data.values:   # ✅ FIXED
       
        seq.append(row)

    seq_df = pd.DataFrame(seq, columns=["volt","rotate","pressure","vibration"])

    seq_scaled = scaler.transform(seq_df)

    seq_scaled = seq_scaled.reshape(1, 30, 4)

    # LSTM prediction
    future_prob = lstm_model.predict(seq_scaled)[0][0]

    final_prob = max(current_prob, future_prob)


    # risk + recommendation
    risk, rec = risk_info(final_prob)

    return health, risk, rec, current_prob, future_prob

# -------------------------
# AUTO MONITORING
# -------------------------

def auto_monitor():
    if SYSTEM_MODE != "AUTO":
      return

    with app.app_context():   # ✅ ADD THIS LINE


        cur=get_cursor()

        cur.execute("SELECT machineID FROM machines")
        machines=cur.fetchall()

        for m in machines:
            machineID=m[0]
            print(f"Processing machine: {machineID}")
            

            cur.execute("""
            SELECT volt,rotate,pressure,vibration
            FROM telemetry
            WHERE machineID=%s
            ORDER BY datetime DESC
            LIMIT 30
            """,(machineID,))

            rows=cur.fetchall()


            if len(rows)<30:
                
                continue

            columns = ["volt","rotate","pressure","vibration"]
            data = pd.DataFrame(rows, columns=columns)

            health,risk,rec,current_prob,future_prob = predict_machine(machineID,data)


            prob = max(current_prob, future_prob)

            cur.execute("""
            INSERT INTO prediction_history
            (user_id,machineID,health_score,risk_level,failure_probability)
            VALUES(%s,%s,%s,%s,%s)
            """,(1,machineID,health,risk,prob))

            mysql.connection.commit()

            
            if risk == "High Risk":
             send_alert(machineID,risk,current_prob,future_prob,rec,health)

        print("✅ Auto Monitoring Executed")
# -------------------------
# HOME
# -------------------------

@app.route("/")
def home():
    return render_template("login.html")

# -------------------------
# REGISTER
# -------------------------
from flask import flash, redirect, url_for,request
@app.route("/register",methods=["POST"])
def register():

    name=request.form["name"]
    email=request.form["email"]
    password=request.form["password"]
    
    cur=get_cursor()
    cur.execute("SELECT * FROM users WHERE email=%s", (email,))
    user = cur.fetchone()

    if user:
        flash("User already registered!", "error")
        return redirect(url_for('home'))
    cur.execute("""
    INSERT INTO users(name,email,password,role,status)
    VALUES(%s,%s,%s,'user','active')
    """,(name,email,password))

    mysql.connection.commit()
    flash("Registered successfully!", "success")
    return redirect(url_for('home'))
    

# -------------------------
# LOGIN
# -------------------------

@app.route("/login",methods=["POST"])
def login():

    email=request.form["email"]
    password=request.form["password"]

    cur=get_cursor()

    cur.execute("""
    SELECT * FROM users
    WHERE email=%s AND password=%s AND status='active'
    """,(email,password))

    user=cur.fetchone()

    if user:

        session["user_id"]=user[0]
        session["role"]=user[4]

        if user[4]=="admin":
            return redirect("/admin_dashboard")

        return redirect("/dashboard")

    return "Login Failed"

# -------------------------
# USER DASHBOARD
# -------------------------

@app.route("/dashboard")
def dashboard():

    res=login_required()
    if res: return res

    cur=get_cursor()

    cur.execute("SELECT machineID FROM machines")
    machines=cur.fetchall()

    return render_template("dashboard.html",machines=machines,mode=SYSTEM_MODE)


# -------------------------
# PREDICTION
# -------------------------
@app.route("/predict", methods=["POST"])
def predict():

    # check login
    res = login_required()
    if res:
        return res

    machineID = request.form.get("machineID")

    # =====================================================
    # MANUAL MODE (NO DATABASE)
    # =====================================================
    if SYSTEM_MODE == "MANUAL":

        try:
            volt = float(request.form.get("volt", 0))
            rotate = float(request.form.get("rotate", 0))
            pressure = float(request.form.get("pressure", 0))
            vibration = float(request.form.get("vibration", 0))

        except:
            return "Invalid input values"

        # validation
        if any(v <= 0 for v in [volt, rotate, pressure, vibration]):
            return "All values must be greater than 0"
        warning = ""

        if not (110 <= volt <= 150):
          warning += "Voltage out of range. "

        if not (250 <= rotate <= 450):
          warning += "Rotation out of range. "

        if not (50 <= pressure <= 100):
           warning += "Pressure out of range. "

        if not (10 <= vibration <= 80):
           warning += "Vibration out of range. "
        try:
            features = [[volt, rotate, pressure, vibration]]
            columns = ["volt", "rotate", "pressure", "vibration"]

            df = pd.DataFrame(features, columns=columns)

            scaled = scaler.transform(df)

            current_prob = xgb_model.predict_proba(scaled)[0][1]

            health = (1 - current_prob) * 100

            risk, rec = risk_info(current_prob)

        except Exception as e:
            return f"Prediction Error: {e}"
        print("WARNING:", warning)
        return render_template(
            "result.html",
            health=round(health, 2),
            risk=risk,
            current_prob=round(current_prob, 3),
            future_prob=None,
            recommendation=rec,
            machineID=machineID,
            mode="MANUAL",
            warning=warning
        )

    # =====================================================
    # AUTO MODE (WITH DATABASE + TELEMETRY)
    # =====================================================
    else:
        try:
            cur = get_cursor()

            # fetch last 30 records
            cur.execute("""
                SELECT volt, rotate, pressure, vibration
                FROM telemetry
                WHERE machineID=%s
                ORDER BY datetime DESC
                LIMIT 30
            """, (machineID,))

            rows = cur.fetchall()

            columns = ["volt", "rotate", "pressure", "vibration"]
            data = pd.DataFrame(rows, columns=columns)

            future_prob = None

            # CASE 0: NO DATA
            if len(rows) == 0:
                return "No telemetry data available for this machine"

            # CASE 1: LESS THAN 30 RECORDS → XGBOOST ONLY
            if len(rows) < 30:

                latest = data.iloc[0].values

                df = pd.DataFrame([latest], columns=columns)

                scaled = scaler.transform(df)

                current_prob = xgb_model.predict_proba(scaled)[0][1]

                health = (1 - current_prob) * 100

                risk, rec = risk_info(current_prob)

            # CASE 2: FULL MODEL
            else:

                health, risk, rec, current_prob, future_prob = predict_machine(
                    machineID, data
                )

            # SAVE RESULT
            cur.execute("""
                INSERT INTO prediction_history
                (user_id, machineID, health_score, risk_level, failure_probability)
                VALUES (%s, %s, %s, %s, %s)
            """, (
                session["user_id"],
                machineID,
                health,
                risk,
                current_prob
            ))

            mysql.connection.commit()

        except Exception as e:
            return f"Auto Prediction Error: {e}"

        return render_template(
            "result.html",
            health=round(health, 2),
            risk=risk,
            current_prob=round(current_prob, 3),
            future_prob=round(future_prob, 3) if future_prob is not None else None,
            recommendation=rec,
            machineID=machineID,
            mode="AUTO"
        )
# TREND GRAPH
# -------------------------

@app.route("/trend/<machineID>")
def trend(machineID):

    res=login_required()
    if res: return res

    cur=get_cursor()

    cur.execute("""
    SELECT health_score,failure_probability,prediction_time
    FROM prediction_history
    WHERE machineID=%s
    """,(machineID,))

    data=cur.fetchall()

    return jsonify(data)

# -------------------------
# HISTORY
# -------------------------

@app.route("/history")
def history():

    res=login_required()
    if res: return res

    cur=get_cursor()

    cur.execute("""
    SELECT machineID,health_score,risk_level,failure_probability,prediction_time
    FROM prediction_history
    ORDER BY prediction_time DESC
    """)

    data=cur.fetchall()

    return render_template("history.html",data=data)

# -------------------------
# ADMIN DASHBOARD
# -------------------------

@app.route("/admin_dashboard")
def admin_dashboard():
    res=login_required()
    if res:return res
    res=admin_required()
    if res: return res

    cur=get_cursor()

    cur.execute("SELECT COUNT(*) FROM machines")
    machines=cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM users")
    users=cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM prediction_history")
    predictions=cur.fetchone()[0]

    return render_template(
        "admin_dashboard.html",
        machines=machines,
        users=users,
        predictions=predictions
    )

# -------------------------
# USER MANAGEMENT
# -------------------------

@app.route("/admin/users")
def admin_users():

    res=admin_required()
    if res: return res

    cur=get_cursor()
    cur.execute("SELECT * FROM users")
    users=cur.fetchall()

    return render_template("users.html",users=users)

@app.route("/admin/delete_user/<id>")
def delete_user(id):

    res=admin_required()
    if res: return res

    cur=get_cursor()
    cur.execute("DELETE FROM users WHERE id=%s",(id,))
    mysql.connection.commit()

    return redirect("/admin/users")

# -------------------------
# MACHINE MANAGEMENT
# -------------------------

@app.route("/admin/machines")
def admin_machines():

    res=admin_required()
    if res: return res

    cur=get_cursor()
    cur.execute("SELECT * FROM machines")
    machines=cur.fetchall()

    return render_template("machines.html",machines=machines)

@app.route("/admin/add_machine",methods=["POST"])
def add_machine():

    res=admin_required()
    if res: return res

    machineID=request.form["machineID"]
    model=request.form["model"]
    age=request.form["age"]

    cur=get_cursor()

    cur.execute("""
    INSERT INTO machines(machineID,model,age)
    VALUES(%s,%s,%s)
    """,(machineID,model,age))

    mysql.connection.commit()

    return redirect("/admin/machines")

@app.route("/admin/delete_machine/<id>")
def delete_machine(id):

    res=admin_required()
    if res: return res

    cur=get_cursor()

    cur.execute("DELETE FROM machines WHERE machineID=%s",(id,))
    mysql.connection.commit()

    return redirect("/admin/machines")

# -------------------------
# MAINTENANCE SCHEDULE
# -------------------------

@app.route("/admin/schedule",methods=["POST"])
def schedule():

    res=admin_required()
    if res: return res

    machineID=request.form["machineID"]
    date=request.form["date"]
    engineer=request.form["engineer"]

    cur=get_cursor()

    cur.execute("""
    INSERT INTO maintenance_schedule(machineID,date,engineer)
    VALUES(%s,%s,%s)
    """,(machineID,date,engineer))

    mysql.connection.commit()
    flash("Maintenace Scheduled Successfully!")
    return redirect("/admin/maintenance_schedule")
# -------------------------
# MAINTENANCE PAGE
# -------------------------

@app.route("/admin/maintenance_schedule")
def maintenance_page():

    res = admin_required()
    if res: return res

    cur = get_cursor()

    cur.execute("SELECT machineID FROM machines")
    machines = cur.fetchall()
    cur.execute(""" SELECT machineID,date,engineer FROM maintenance_schedule ORDER BY date DESC""")
    schedules=cur.fetchall()
    return render_template("maintenance_schedule.html", machines=machines,schedules=schedules)
# -------------------------
# LOGOUT
# -------------------------

@app.route("/logout")
def logout():

    session.clear()

    return redirect("/")

# -------------------------
# AUTO MONITOR SCHEDULER
# -------------------------
scheduler = BackgroundScheduler() 
# telemetry every 10 sec
scheduler.add_job(
    func=generate_simulated_data,
    trigger="interval",
    seconds=10
)


# prediction every 1 min
scheduler.add_job(
    func=auto_monitor,
    trigger="interval",
    minutes=1,
    max_instances=1
)

# -------------------------

if __name__=="__main__":
    scheduler.start()
    app.run(debug=False)