from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from codenames import get_clue, get_analysis

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")


templates = Jinja2Templates(directory="templates")


@app.post("/get_clue", response_class=JSONResponse)
async def read_item(request: Request, 
	red_words: str = 'boom,puppet,scarecrow,nutcracker,turn,door,victory,whistle', 
	blue_words: str = 'pool,scarf,comet,sled,port,bunk,ribbon,rock,ninja', 
	neutral_words: str = 'park,pad,pupil,goldilocks,tile,columbus,fiddle', 
	assassin_word: str = 'space',
	positive_color: str = 'red', 
	picked_words: str = 'n/a',
	primary_strategy: str='max_n', 
        secondary_strategy: str='max_total_similarity',
        minimum_similarity: float = 0.0, #.315,
        min_diff_from_neutral: float = 0.0,
        min_diff_from_negative: float = 0.05, 
        min_diff_from_assassin: float = 0.07):
    code_word, count = get_clue(board={
			   'positive': red_words.split(',') if positive_color=='red' else blue_words.split(','),
			   'negative': blue_words.split(',') if positive_color=='red' else red_words.split(','),
			   'neutral': neutral_words.split(','),
			   'assassin': assassin_word.split(',')
			  },
		    picked_words=picked_words.split(','),
			primary_strategy=primary_strategy, 
                        secondary_strategy=secondary_strategy,
                        minimum_similarity=minimum_similarity,
                        min_diff_from_neutral=min_diff_from_neutral,
                        min_diff_from_negative=min_diff_from_negative, 
                        min_diff_from_assassin=min_diff_from_assassin
		    )
    return {'code_word': code_word, 'number': count}


@app.post("/analyze_clue", response_class=JSONResponse)
async def read_item(request: Request, 
	code_word: str = 'deck',
	red_words: str = 'boom,puppet,scarecrow,nutcracker,turn,door,victory,whistle', 
	blue_words: str = 'pool,scarf,comet,sled,port,bunk,ribbon,rock,ninja', 
	neutral_words: str = 'park,pad,pupil,goldilocks,tile,columbus,fiddle', 
	assassin_word: str = 'space',
	positive_color: str = 'red', 
	picked_words: str = 'n/a'):
    analysis = get_analysis(code_word, board={'positive': red_words.split(',') if positive_color=='red' else blue_words.split(','),
			   'negative': blue_words.split(',') if positive_color=='red' else red_words.split(','),
			   'neutral': neutral_words.split(','),
			   'assassin': assassin_word.split(',')
			  },
		    picked_words=picked_words.split(',')
		    ).to_dict(orient='records')
    return analysis


@app.post("/get_and_analyze_clue", response_class=JSONResponse)
async def read_item(request: Request, 
	red_words: str = 'boom,puppet,scarecrow,nutcracker,turn,door,victory,whistle', 
	blue_words: str = 'pool,scarf,comet,sled,port,bunk,ribbon,rock,ninja',
	neutral_words: str = 'park,pad,pupil,goldilocks,tile,columbus,fiddle', 
	assassin_word: str = 'space',
	positive_color: str = 'red', 
	picked_words: str = 'n/a',
	primary_strategy: str='max_n', 
        secondary_strategy: str='max_total_similarity',
        minimum_similarity: float=0.0, #.315,
        min_diff_from_neutral: float=0.0,
        min_diff_from_negative: float=0.05, #0.01, 
        min_diff_from_assassin: float=0.07): #0.05
    code_word, count = get_clue(board={
			   'positive': red_words.split(',') if positive_color=='red' else blue_words.split(','),
			   'negative': blue_words.split(',') if positive_color=='red' else red_words.split(','),
			   'neutral': neutral_words.split(','),
			   'assassin': assassin_word.split(',')
			  },
		    picked_words=picked_words.split(','),
			primary_strategy=primary_strategy, 
                        secondary_strategy=secondary_strategy,
                        minimum_similarity=minimum_similarity,
                        min_diff_from_neutral=min_diff_from_neutral,
                        min_diff_from_negative=min_diff_from_negative, 
                        min_diff_from_assassin=min_diff_from_assassin
		    )
    analysis = get_analysis(code_word, board={'positive': red_words.split(',') if positive_color=='red' else blue_words.split(','),
			   'negative': blue_words.split(',') if positive_color=='red' else red_words.split(','),
			   'neutral': neutral_words.split(','),
			   'assassin': assassin_word.split(',')
			  },
		    picked_words=picked_words.split(',')
		    ).to_dict(orient='records')
    return {'code_word': code_word, 'number': count, 'analysis': analysis}