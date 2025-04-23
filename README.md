# MacroScope
A full-stack quantitative finance project meant to analyze global events/relevant macro-economic trends.

04.23.2025
## Project Set-Up
Use the following command to create a virtual env called macroVenv, or name it what you'd like: python3 -m venv macroVenv
For good practice, I set up my venv in a separate folder outside called virtual-envs

Replace line 4 in the run.sh file with your path and venv name
Run the following line once: chmod +x run.sh
Now to run the program, simply enter bash run.sh in your terminal!

This will download all packages from requirements.txt and run the recession_tool.py script automatically. 

## Config File
In config.toml, please replace the FRED API key and Reddit scraping dev keys with your own. For guidance, please visit [the FRED site] (https://fred.stlouisfed.org/docs/api/api_key.html) and after creating a Reddit profile, [this Reddit page] (https://www.reddit.com/prefs/apps). 

## Future Additions
Under the pages folder, add .py files in this format "X__Name.py", where X is the subsequent number of the new subpage.
