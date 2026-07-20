# esg_score.py


def calculate_esg_score(

        pvi,

        carbon_g

):


    score = 100



    # 과포장 패널티

    if pvi >=80:

        score -=40


    elif pvi >=60:

        score -=30


    elif pvi >=40:

        score -=15




    # 탄소 패널티

    if carbon_g >=500:

        score -=20


    elif carbon_g >=300:

        score -=10



    # 점수 보정

    if score <0:

        score=0



    if score>=90:

        grade="S"


    elif score>=80:

        grade="A"


    elif score>=70:

        grade="B"


    elif score>=60:

        grade="C"


    else:

        grade="D"



    return {


        "score":

        score,


        "grade":

        grade

    }
