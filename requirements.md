requirements.md

Req 1
As a tennis team captain I want to get AI generated insights about our next tennis match opponents. 

Req 2
The AI should compare the player ratings between the two teams partly by pulling it from tennisrecord.com. This is an example url to our current team https://www.tennisrecord.com/adult/teamprofile.aspx?teamname=RBW-Long%20Shots-Ferre&year=2026&s=2
From above URL the AI can find the current estimated rating from all players in our team, and find the URL (example https://www.tennisrecord.com/adult/league.aspx?subflightname=A&year=2026&s=2) to the current leagure with other team pages. 

Req 3
The AI should be able to get data from tennislink.usta.com (example team URL https://tennislink.usta.com/Leagues/Main/StatsAndStandings.aspx?t=R-3#&&s=w824NFtIC_FAPNS0UYIp9SmAnfF6QRK40) and identify the next match based on todays date. 

Req 4
The captain should be able to give a tentative line-up for our team, and then the AI should use that as input together with the data pulled to predict each court result. 

Req 5
The input and output should be done through a python CLI command, with the match predictions also available as a .csv file if the user chooses that. 

Req 6
tennislink.usta.com requires the user to add credentials to the AI agent, these credentials must remain on the local machine only and never shared in any way in the GitHub repo. 

Req 7 
The backend code should be written with Python

Req 8 
The requirements should be done in a way so it's an AI agent we have created. 

Req 9
When the AI makes the match outcome predictions I want the AI to consider (but not limited to):
1. The rating from tennisrecord.com
2. The trend of both teams for each court, if opponents have won Singles 1 everytime that should count into the prediciton
3. win/loose streaks for teams and players, check player history for matches outside of this team. Consider matches up to 6 months before todays match. 
4. an prediction of opponents for all courts on the oposing team
5. I want the AI to review the tentative line-up give suggestions on changes to maximize the chances of winning the match.
6. The highest priority should be to win the team match and not any individual courts.
7. Consider wins and losses on higher courts to be more difficult (example weigh a court 1 win higher than a court 3 win)  

Req 10
check the tennislink for the match format. Normally it's always 3 doubles, and some matches are only 1 singles and some are 2 singles

