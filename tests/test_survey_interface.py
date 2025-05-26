from mjtracker import SurveysInterface, SurveyInterface


def test_import_and_play_with_interface():
    si = SurveysInterface.load("/home/pierre/Documents/Mieux_voter/database-mj-2027/mj-database-2027/mj2027.csv")
    a_survey = si.surveys[0]
    si._intentions_colheaders
    one_survey = si.select_survey(a_survey)
    print(one_survey)

    one_survey._sanity_check_on_intentions()

    one_survey.df["poll_id"]
    one_survey.nb_surveys
    one_survey.nb_grades
    one_survey._intentions_colheaders
    one_survey._intentions_without_no_opinion_colheaders

    one_survey._no_opinion_colheader
    one_survey._intention_no_opinion_colheader
    one_survey._grades_colheaders

    one_survey.total_intentions
    one_survey.total_intentions_no_opinion

    one_survey.total_intentions_without_no_opinion

    yeah = one_survey.to_no_opinion_survey()  # ca marche pas encore

    one_survey.intentions  # ca marche pas encore
    one_survey.candidates  # ca marche pas encore

    si.surveys
    si._intentions_colheaders
    si.select_survey(si.surveys[0])
    si.to_no_opinion_surveys()
