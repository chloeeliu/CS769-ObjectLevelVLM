import json
import os
import torch
import numpy as np
import pycocotools.mask as mask
import math
import skimage
import pickle
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset
from tqdm import tqdm

class LVISDataset(Dataset):
    # Path to instances.json
    # Path to COCO2014/COCO2017 train images
    def __init__(self, config, split="train", patch_size=16):
        super(LVISDataset, self).__init__()
        self.split = split
        self.grid_size=patch_size
        self.class_counts = {}
        dataset_path = f"/data/ossowski/COCO2017/LVIS/{split}/lvis_v1_{split}.json"
        self.data = json.load(open(dataset_path))
        self.entries = self._load_dataset()

        if config["use_retrieval"]:
            if config["crop_image"]:
                cropped = "cropped_"
            else:
                cropped = ""
            CACHE_PATH = f'cache/{config["task"]}/retrieval_cache_{config["examples_per_class"]}_{cropped}{config["vision_encoder"].split("/")[-1]}.pkl'
            if os.path.exists(CACHE_PATH):
                with open(CACHE_PATH, 'rb') as f:
                    self.cache = pickle.load(f)
                    print(f"Loaded cached retrieval similarity from {CACHE_PATH}")

    def _get_ViT_mask(self, segmentation, height, width):
        
        output_width, output_height = self.grid_size, self.grid_size
        rles = mask.frPyObjects(segmentation, height, width)
        rle = mask.merge(rles)
        m =  mask.decode(rle).astype(bool)
        

        rle['counts'] = rle['counts'].decode('ascii')
        pooled_mask = skimage.measure.block_reduce(m, block_size=(math.floor(height / self.grid_size), math.floor(width / self.grid_size)), func=np.max)
        # if (np.sum(pooled_mask, axis=None) == 0):
        #     print(m.shape)
        #     print(np.sum(m, axis = None))
        # assert np.sum(pooled_mask, axis=None) > 0

        result_height, result_width = pooled_mask.shape

        # If the result is smaller than 16x16, pad it with zeros (but not many images are less than 16 in one dimension)
        if result_height < output_height or result_width < output_width:
            pad_height = output_height - result_height
            pad_width = output_width - result_width
            pooled_mask = np.pad(pooled_mask, ((0, pad_height), (0, pad_width)), mode='constant')

        if result_height > output_height:
            merged = np.max(np.concatenate((pooled_mask[-1:, :], pooled_mask[-2:-1, :]), axis = 0), axis = 0)
            pooled_mask = pooled_mask[:output_height, :]
            pooled_mask[-1, :] = merged
        if result_width > output_width:
            merged = np.max(np.concatenate((pooled_mask[:, -1:], pooled_mask[:, -2:-1]), axis = 1), axis = 1)
            pooled_mask = pooled_mask[:, :output_width]
            pooled_mask[:, -1] = merged

        assert pooled_mask.shape == (output_height,output_width)
        return pooled_mask
    
    # Should take in argument k: how many closest objects to retrieve
    # and object features: the ViT features of query objects
    # Return: The k closest examples from self.entries
    def retrieve_closest(self, object_features, k, train_phase = True, b_num=0):

        if self.cache != None:
            most_relevant = self.cache[b_num][:k]
            similarity_scores = [[round(x[1], 2) for x in most_relevant]]
            retrieved_info = [[self.entries[x[0]] for x in most_relevant]]
        else:
            dist_matrix = (object_features @ self.retrieval_keys.T)
            if train_phase:
                closest_indices = torch.argsort(dist_matrix, axis = -1, descending=True)[:, 1:1+k]
            else:
                closest_indices = torch.argsort(dist_matrix, axis = -1, descending=True)[:, 0:k]
                similarity_scores = [[round(dist_matrix[i, x].item(), 2) for x in closest_indices[i,:]] for i in range(len(closest_indices))]

                retrieved_info = [[self.entries[x] for x in closest_indices[i,:]] for i in range(len(closest_indices))]
        return retrieved_info, similarity_scores
        

    def _load_dataset(self):

        CLASSES = [
        'aerosol_can', 'air_conditioner', 'airplane', 'alarm_clock', 'alcohol',
        'alligator', 'almond', 'ambulance', 'amplifier', 'anklet', 'antenna',
        'apple', 'applesauce', 'apricot', 'apron', 'aquarium',
        'arctic_(type_of_shoe)', 'armband', 'armchair', 'armoire', 'armor',
        'artichoke', 'trash_can', 'ashtray', 'asparagus', 'atomizer',
        'avocado', 'award', 'awning', 'ax', 'baboon', 'baby_buggy',
        'basketball_backboard', 'backpack', 'handbag', 'suitcase', 'bagel',
        'bagpipe', 'baguet', 'bait', 'ball', 'ballet_skirt', 'balloon',
        'bamboo', 'banana', 'Band_Aid', 'bandage', 'bandanna', 'banjo',
        'banner', 'barbell', 'barge', 'barrel', 'barrette', 'barrow',
        'baseball_base', 'baseball', 'baseball_bat', 'baseball_cap',
        'baseball_glove', 'basket', 'basketball', 'bass_horn', 'bat_(animal)',
        'bath_mat', 'bath_towel', 'bathrobe', 'bathtub', 'batter_(food)',
        'battery', 'beachball', 'bead', 'bean_curd', 'beanbag', 'beanie',
        'bear', 'bed', 'bedpan', 'bedspread', 'cow', 'beef_(food)', 'beeper',
        'beer_bottle', 'beer_can', 'beetle', 'bell', 'bell_pepper', 'belt',
        'belt_buckle', 'bench', 'beret', 'bib', 'Bible', 'bicycle', 'visor',
        'billboard', 'binder', 'binoculars', 'bird', 'birdfeeder', 'birdbath',
        'birdcage', 'birdhouse', 'birthday_cake', 'birthday_card',
        'pirate_flag', 'black_sheep', 'blackberry', 'blackboard', 'blanket',
        'blazer', 'blender', 'blimp', 'blinker', 'blouse', 'blueberry',
        'gameboard', 'boat', 'bob', 'bobbin', 'bobby_pin', 'boiled_egg',
        'bolo_tie', 'deadbolt', 'bolt', 'bonnet', 'book', 'bookcase',
        'booklet', 'bookmark', 'boom_microphone', 'boot', 'bottle',
        'bottle_opener', 'bouquet', 'bow_(weapon)', 'bow_(decorative_ribbons)',
        'bow-tie', 'bowl', 'pipe_bowl', 'bowler_hat', 'bowling_ball', 'box',
        'boxing_glove', 'suspenders', 'bracelet', 'brass_plaque', 'brassiere',
        'bread-bin', 'bread', 'breechcloth', 'bridal_gown', 'briefcase',
        'broccoli', 'broach', 'broom', 'brownie', 'brussels_sprouts',
        'bubble_gum', 'bucket', 'horse_buggy', 'bull', 'bulldog', 'bulldozer',
        'bullet_train', 'bulletin_board', 'bulletproof_vest', 'bullhorn',
        'bun', 'bunk_bed', 'buoy', 'burrito', 'bus_(vehicle)', 'business_card',
        'butter', 'butterfly', 'button', 'cab_(taxi)', 'cabana', 'cabin_car',
        'cabinet', 'locker', 'cake', 'calculator', 'calendar', 'calf',
        'camcorder', 'camel', 'camera', 'camera_lens', 'camper_(vehicle)',
        'can', 'can_opener', 'candle', 'candle_holder', 'candy_bar',
        'candy_cane', 'walking_cane', 'canister', 'canoe', 'cantaloup',
        'canteen', 'cap_(headwear)', 'bottle_cap', 'cape', 'cappuccino',
        'car_(automobile)', 'railcar_(part_of_a_train)', 'elevator_car',
        'car_battery', 'identity_card', 'card', 'cardigan', 'cargo_ship',
        'carnation', 'horse_carriage', 'carrot', 'tote_bag', 'cart', 'carton',
        'cash_register', 'casserole', 'cassette', 'cast', 'cat', 'cauliflower',
        'cayenne_(spice)', 'CD_player', 'celery', 'cellular_telephone',
        'chain_mail', 'chair', 'chaise_longue', 'chalice', 'chandelier',
        'chap', 'checkbook', 'checkerboard', 'cherry', 'chessboard',
        'chicken_(animal)', 'chickpea', 'chili_(vegetable)', 'chime',
        'chinaware', 'crisp_(potato_chip)', 'poker_chip', 'chocolate_bar',
        'chocolate_cake', 'chocolate_milk', 'chocolate_mousse', 'choker',
        'chopping_board', 'chopstick', 'Christmas_tree', 'slide', 'cider',
        'cigar_box', 'cigarette', 'cigarette_case', 'cistern', 'clarinet',
        'clasp', 'cleansing_agent', 'cleat_(for_securing_rope)', 'clementine',
        'clip', 'clipboard', 'clippers_(for_plants)', 'cloak', 'clock',
        'clock_tower', 'clothes_hamper', 'clothespin', 'clutch_bag', 'coaster',
        'coat', 'coat_hanger', 'coatrack', 'cock', 'cockroach',
        'cocoa_(beverage)', 'coconut', 'coffee_maker', 'coffee_table',
        'coffeepot', 'coil', 'coin', 'colander', 'coleslaw',
        'coloring_material', 'combination_lock', 'pacifier', 'comic_book',
        'compass', 'computer_keyboard', 'condiment', 'cone', 'control',
        'convertible_(automobile)', 'sofa_bed', 'cooker', 'cookie',
        'cooking_utensil', 'cooler_(for_food)', 'cork_(bottle_plug)',
        'corkboard', 'corkscrew', 'edible_corn', 'cornbread', 'cornet',
        'cornice', 'cornmeal', 'corset', 'costume', 'cougar', 'coverall',
        'cowbell', 'cowboy_hat', 'crab_(animal)', 'crabmeat', 'cracker',
        'crape', 'crate', 'crayon', 'cream_pitcher', 'crescent_roll', 'crib',
        'crock_pot', 'crossbar', 'crouton', 'crow', 'crowbar', 'crown',
        'crucifix', 'cruise_ship', 'police_cruiser', 'crumb', 'crutch',
        'cub_(animal)', 'cube', 'cucumber', 'cufflink', 'cup', 'trophy_cup',
        'cupboard', 'cupcake', 'hair_curler', 'curling_iron', 'curtain',
        'cushion', 'cylinder', 'cymbal', 'dagger', 'dalmatian', 'dartboard',
        'date_(fruit)', 'deck_chair', 'deer', 'dental_floss', 'desk',
        'detergent', 'diaper', 'diary', 'die', 'dinghy', 'dining_table', 'tux',
        'dish', 'dish_antenna', 'dishrag', 'dishtowel', 'dishwasher',
        'dishwasher_detergent', 'dispenser', 'diving_board', 'Dixie_cup',
        'dog', 'dog_collar', 'doll', 'dollar', 'dollhouse', 'dolphin',
        'domestic_ass', 'doorknob', 'doormat', 'doughnut', 'dove', 'dragonfly',
        'drawer', 'underdrawers', 'dress', 'dress_hat', 'dress_suit',
        'dresser', 'drill', 'drone', 'dropper', 'drum_(musical_instrument)',
        'drumstick', 'duck', 'duckling', 'duct_tape', 'duffel_bag', 'dumbbell',
        'dumpster', 'dustpan', 'eagle', 'earphone', 'earplug', 'earring',
        'easel', 'eclair', 'eel', 'egg', 'egg_roll', 'egg_yolk', 'eggbeater',
        'eggplant', 'electric_chair', 'refrigerator', 'elephant', 'elk',
        'envelope', 'eraser', 'escargot', 'eyepatch', 'falcon', 'fan',
        'faucet', 'fedora', 'ferret', 'Ferris_wheel', 'ferry', 'fig_(fruit)',
        'fighter_jet', 'figurine', 'file_cabinet', 'file_(tool)', 'fire_alarm',
        'fire_engine', 'fire_extinguisher', 'fire_hose', 'fireplace',
        'fireplug', 'first-aid_kit', 'fish', 'fish_(food)', 'fishbowl',
        'fishing_rod', 'flag', 'flagpole', 'flamingo', 'flannel', 'flap',
        'flash', 'flashlight', 'fleece', 'flip-flop_(sandal)',
        'flipper_(footwear)', 'flower_arrangement', 'flute_glass', 'foal',
        'folding_chair', 'food_processor', 'football_(American)',
        'football_helmet', 'footstool', 'fork', 'forklift', 'freight_car',
        'French_toast', 'freshener', 'frisbee', 'frog', 'fruit_juice',
        'frying_pan', 'fudge', 'funnel', 'futon', 'gag', 'garbage',
        'garbage_truck', 'garden_hose', 'gargle', 'gargoyle', 'garlic',
        'gasmask', 'gazelle', 'gelatin', 'gemstone', 'generator',
        'giant_panda', 'gift_wrap', 'ginger', 'giraffe', 'cincture',
        'glass_(drink_container)', 'globe', 'glove', 'goat', 'goggles',
        'goldfish', 'golf_club', 'golfcart', 'gondola_(boat)', 'goose',
        'gorilla', 'gourd', 'grape', 'grater', 'gravestone', 'gravy_boat',
        'green_bean', 'green_onion', 'griddle', 'grill', 'grits', 'grizzly',
        'grocery_bag', 'guitar', 'gull', 'gun', 'hairbrush', 'hairnet',
        'hairpin', 'halter_top', 'ham', 'hamburger', 'hammer', 'hammock',
        'hamper', 'hamster', 'hair_dryer', 'hand_glass', 'hand_towel',
        'handcart', 'handcuff', 'handkerchief', 'handle', 'handsaw',
        'hardback_book', 'harmonium', 'hat', 'hatbox', 'veil', 'headband',
        'headboard', 'headlight', 'headscarf', 'headset',
        'headstall_(for_horses)', 'heart', 'heater', 'helicopter', 'helmet',
        'heron', 'highchair', 'hinge', 'hippopotamus', 'hockey_stick', 'hog',
        'home_plate_(baseball)', 'honey', 'fume_hood', 'hook', 'hookah',
        'hornet', 'horse', 'hose', 'hot-air_balloon', 'hotplate', 'hot_sauce',
        'hourglass', 'houseboat', 'hummingbird', 'hummus', 'polar_bear',
        'icecream', 'popsicle', 'ice_maker', 'ice_pack', 'ice_skate',
        'igniter', 'inhaler', 'iPod', 'iron_(for_clothing)', 'ironing_board',
        'jacket', 'jam', 'jar', 'jean', 'jeep', 'jelly_bean', 'jersey',
        'jet_plane', 'jewel', 'jewelry', 'joystick', 'jumpsuit', 'kayak',
        'keg', 'kennel', 'kettle', 'key', 'keycard', 'kilt', 'kimono',
        'kitchen_sink', 'kitchen_table', 'kite', 'kitten', 'kiwi_fruit',
        'knee_pad', 'knife', 'knitting_needle', 'knob', 'knocker_(on_a_door)',
        'koala', 'lab_coat', 'ladder', 'ladle', 'ladybug', 'lamb_(animal)',
        'lamb-chop', 'lamp', 'lamppost', 'lampshade', 'lantern', 'lanyard',
        'laptop_computer', 'lasagna', 'latch', 'lawn_mower', 'leather',
        'legging_(clothing)', 'Lego', 'legume', 'lemon', 'lemonade', 'lettuce',
        'license_plate', 'life_buoy', 'life_jacket', 'lightbulb',
        'lightning_rod', 'lime', 'limousine', 'lion', 'lip_balm', 'liquor',
        'lizard', 'log', 'lollipop', 'speaker_(stereo_equipment)', 'loveseat',
        'machine_gun', 'magazine', 'magnet', 'mail_slot', 'mailbox_(at_home)',
        'mallard', 'mallet', 'mammoth', 'manatee', 'mandarin_orange', 'manger',
        'manhole', 'map', 'marker', 'martini', 'mascot', 'mashed_potato',
        'masher', 'mask', 'mast', 'mat_(gym_equipment)', 'matchbox',
        'mattress', 'measuring_cup', 'measuring_stick', 'meatball', 'medicine',
        'melon', 'microphone', 'microscope', 'microwave_oven', 'milestone',
        'milk', 'milk_can', 'milkshake', 'minivan', 'mint_candy', 'mirror',
        'mitten', 'mixer_(kitchen_tool)', 'money',
        'monitor_(computer_equipment) computer_monitor', 'monkey', 'motor',
        'motor_scooter', 'motor_vehicle', 'motorcycle', 'mound_(baseball)',
        'mouse_(computer_equipment)', 'mousepad', 'muffin', 'mug', 'mushroom',
        'music_stool', 'musical_instrument', 'nailfile', 'napkin',
        'neckerchief', 'necklace', 'necktie', 'needle', 'nest', 'newspaper',
        'newsstand', 'nightshirt', 'nosebag_(for_animals)',
        'noseband_(for_animals)', 'notebook', 'notepad', 'nut', 'nutcracker',
        'oar', 'octopus_(food)', 'octopus_(animal)', 'oil_lamp', 'olive_oil',
        'omelet', 'onion', 'orange_(fruit)', 'orange_juice', 'ostrich',
        'ottoman', 'oven', 'overalls_(clothing)', 'owl', 'packet', 'inkpad',
        'pad', 'paddle', 'padlock', 'paintbrush', 'painting', 'pajamas',
        'palette', 'pan_(for_cooking)', 'pan_(metal_container)', 'pancake',
        'pantyhose', 'papaya', 'paper_plate', 'paper_towel', 'paperback_book',
        'paperweight', 'parachute', 'parakeet', 'parasail_(sports)', 'parasol',
        'parchment', 'parka', 'parking_meter', 'parrot',
        'passenger_car_(part_of_a_train)', 'passenger_ship', 'passport',
        'pastry', 'patty_(food)', 'pea_(food)', 'peach', 'peanut_butter',
        'pear', 'peeler_(tool_for_fruit_and_vegetables)', 'wooden_leg',
        'pegboard', 'pelican', 'pen', 'pencil', 'pencil_box',
        'pencil_sharpener', 'pendulum', 'penguin', 'pennant', 'penny_(coin)',
        'pepper', 'pepper_mill', 'perfume', 'persimmon', 'person', 'pet',
        'pew_(church_bench)', 'phonebook', 'phonograph_record', 'piano',
        'pickle', 'pickup_truck', 'pie', 'pigeon', 'piggy_bank', 'pillow',
        'pin_(non_jewelry)', 'pineapple', 'pinecone', 'ping-pong_ball',
        'pinwheel', 'tobacco_pipe', 'pipe', 'pistol', 'pita_(bread)',
        'pitcher_(vessel_for_liquid)', 'pitchfork', 'pizza', 'place_mat',
        'plate', 'platter', 'playpen', 'pliers', 'plow_(farm_equipment)',
        'plume', 'pocket_watch', 'pocketknife', 'poker_(fire_stirring_tool)',
        'pole', 'polo_shirt', 'poncho', 'pony', 'pool_table', 'pop_(soda)',
        'postbox_(public)', 'postcard', 'poster', 'pot', 'flowerpot', 'potato',
        'potholder', 'pottery', 'pouch', 'power_shovel', 'prawn', 'pretzel',
        'printer', 'projectile_(weapon)', 'projector', 'propeller', 'prune',
        'pudding', 'puffer_(fish)', 'puffin', 'pug-dog', 'pumpkin', 'puncher',
        'puppet', 'puppy', 'quesadilla', 'quiche', 'quilt', 'rabbit',
        'race_car', 'racket', 'radar', 'radiator', 'radio_receiver', 'radish',
        'raft', 'rag_doll', 'raincoat', 'ram_(animal)', 'raspberry', 'rat',
        'razorblade', 'reamer_(juicer)', 'rearview_mirror', 'receipt',
        'recliner', 'record_player', 'reflector', 'remote_control',
        'rhinoceros', 'rib_(food)', 'rifle', 'ring', 'river_boat', 'road_map',
        'robe', 'rocking_chair', 'rodent', 'roller_skate', 'Rollerblade',
        'rolling_pin', 'root_beer', 'router_(computer_equipment)',
        'rubber_band', 'runner_(carpet)', 'plastic_bag',
        'saddle_(on_an_animal)', 'saddle_blanket', 'saddlebag', 'safety_pin',
        'sail', 'salad', 'salad_plate', 'salami', 'salmon_(fish)',
        'salmon_(food)', 'salsa', 'saltshaker', 'sandal_(type_of_shoe)',
        'sandwich', 'satchel', 'saucepan', 'saucer', 'sausage', 'sawhorse',
        'saxophone', 'scale_(measuring_instrument)', 'scarecrow', 'scarf',
        'school_bus', 'scissors', 'scoreboard', 'scraper', 'screwdriver',
        'scrubbing_brush', 'sculpture', 'seabird', 'seahorse', 'seaplane',
        'seashell', 'sewing_machine', 'shaker', 'shampoo', 'shark',
        'sharpener', 'Sharpie', 'shaver_(electric)', 'shaving_cream', 'shawl',
        'shears', 'sheep', 'shepherd_dog', 'sherbert', 'shield', 'shirt',
        'shoe', 'shopping_bag', 'shopping_cart', 'short_pants', 'shot_glass',
        'shoulder_bag', 'shovel', 'shower_head', 'shower_cap',
        'shower_curtain', 'shredder_(for_paper)', 'signboard', 'silo', 'sink',
        'skateboard', 'skewer', 'ski', 'ski_boot', 'ski_parka', 'ski_pole',
        'skirt', 'skullcap', 'sled', 'sleeping_bag', 'sling_(bandage)',
        'slipper_(footwear)', 'smoothie', 'snake', 'snowboard', 'snowman',
        'snowmobile', 'soap', 'soccer_ball', 'sock', 'sofa', 'softball',
        'solar_array', 'sombrero', 'soup', 'soup_bowl', 'soupspoon',
        'sour_cream', 'soya_milk', 'space_shuttle', 'sparkler_(fireworks)',
        'spatula', 'spear', 'spectacles', 'spice_rack', 'spider', 'crawfish',
        'sponge', 'spoon', 'sportswear', 'spotlight', 'squid_(food)',
        'squirrel', 'stagecoach', 'stapler_(stapling_machine)', 'starfish',
        'statue_(sculpture)', 'steak_(food)', 'steak_knife', 'steering_wheel',
        'stepladder', 'step_stool', 'stereo_(sound_system)', 'stew', 'stirrer',
        'stirrup', 'stool', 'stop_sign', 'brake_light', 'stove', 'strainer',
        'strap', 'straw_(for_drinking)', 'strawberry', 'street_sign',
        'streetlight', 'string_cheese', 'stylus', 'subwoofer', 'sugar_bowl',
        'sugarcane_(plant)', 'suit_(clothing)', 'sunflower', 'sunglasses',
        'sunhat', 'surfboard', 'sushi', 'mop', 'sweat_pants', 'sweatband',
        'sweater', 'sweatshirt', 'sweet_potato', 'swimsuit', 'sword',
        'syringe', 'Tabasco_sauce', 'table-tennis_table', 'table',
        'table_lamp', 'tablecloth', 'tachometer', 'taco', 'tag', 'taillight',
        'tambourine', 'army_tank', 'tank_(storage_vessel)',
        'tank_top_(clothing)', 'tape_(sticky_cloth_or_paper)', 'tape_measure',
        'tapestry', 'tarp', 'tartan', 'tassel', 'tea_bag', 'teacup',
        'teakettle', 'teapot', 'teddy_bear', 'telephone', 'telephone_booth',
        'telephone_pole', 'telephoto_lens', 'television_camera',
        'television_set', 'tennis_ball', 'tennis_racket', 'tequila',
        'thermometer', 'thermos_bottle', 'thermostat', 'thimble', 'thread',
        'thumbtack', 'tiara', 'tiger', 'tights_(clothing)', 'timer', 'tinfoil',
        'tinsel', 'tissue_paper', 'toast_(food)', 'toaster', 'toaster_oven',
        'toilet', 'toilet_tissue', 'tomato', 'tongs', 'toolbox', 'toothbrush',
        'toothpaste', 'toothpick', 'cover', 'tortilla', 'tow_truck', 'towel',
        'towel_rack', 'toy', 'tractor_(farm_equipment)', 'traffic_light',
        'dirt_bike', 'trailer_truck', 'train_(railroad_vehicle)', 'trampoline',
        'tray', 'trench_coat', 'triangle_(musical_instrument)', 'tricycle',
        'tripod', 'trousers', 'truck', 'truffle_(chocolate)', 'trunk', 'vat',
        'turban', 'turkey_(food)', 'turnip', 'turtle', 'turtleneck_(clothing)',
        'typewriter', 'umbrella', 'underwear', 'unicycle', 'urinal', 'urn',
        'vacuum_cleaner', 'vase', 'vending_machine', 'vent', 'vest',
        'videotape', 'vinegar', 'violin', 'vodka', 'volleyball', 'vulture',
        'waffle', 'waffle_iron', 'wagon', 'wagon_wheel', 'walking_stick',
        'wall_clock', 'wall_socket', 'wallet', 'walrus', 'wardrobe',
        'washbasin', 'automatic_washer', 'watch', 'water_bottle',
        'water_cooler', 'water_faucet', 'water_heater', 'water_jug',
        'water_gun', 'water_scooter', 'water_ski', 'water_tower',
        'watering_can', 'watermelon', 'weathervane', 'webcam', 'wedding_cake',
        'wedding_ring', 'wet_suit', 'wheel', 'wheelchair', 'whipped_cream',
        'whistle', 'wig', 'wind_chime', 'windmill', 'window_box_(for_plants)',
        'windshield_wiper', 'windsock', 'wine_bottle', 'wine_bucket',
        'wineglass', 'blinder_(for_horses)', 'wok', 'wolf', 'wooden_spoon',
        'wreath', 'wrench', 'wristband', 'wristlet', 'yacht', 'yogurt',
        'yoke_(animal_equipment)', 'zebra', 'zucchini']

        classes = [x.replace('_', ' ') for x in CLASSES]

        entries = []
        skipped = 0
        for ann in tqdm(self.data['annotations']):
            label = classes[ann['category_id'] - 1]
            extension = str(ann['image_id']).zfill(12) + ".jpg"
            path_to_image = os.path.join(f"/data/ossowski/COCO2017/{self.split}/data/", extension)
            if not os.path.exists(os.path.join(f"/data/ossowski/COCO2017/{self.split}/data/", extension)):
                path_to_image = os.path.join(f"/data/ossowski/COCO2017/train/data/", extension)
            image = Image.open(path_to_image)
            seg = np.append(1, self._get_ViT_mask(ann['segmentation'], image.height, image.width).flatten())
            rles = mask.frPyObjects(ann['segmentation'], image.height, image.width)
            rle = mask.merge(rles)
            if sum(seg[1:]) == 0:
                skipped += 1
                continue
            if label not in self.class_counts:
                self.class_counts[label] = 0
            self.class_counts[label] += 1
            entries.append({"id": ann["id"], 
                            "path_to_image": path_to_image, 
                            "question": "[obj] What is this?", 
                            "vit_mask": seg, "answer": label, 
                            "original_segmentation": rle, 
                            'bbox':ann['bbox']})

        print(f"Skipped {skipped} entries")
        return entries
    
    def collate_fn(self, batch):
        return {
            "id": [item["id"] for item in batch],
            'path_to_image': [item["path_to_image"] for item in batch],
            'question': [item["question"] for item in batch],
            'vit_mask': [item["vit_mask"] for item in batch],
            "answer":  [item["answer"] for item in batch],
            "original_segmentation":  [item["original_segmentation"] for item in batch],
            'bbox': [item["bbox"] for item in batch]
        }

    def eval_correctness(self, prediction, answer):
        correct = 1 if prediction.lower() == answer.lower() else 0
        score_dict = {}
        score_dict["score"] = correct
        return  score_dict

    def stats(self):
        class_counts = {}
        for example in self.entries:
            c = example['answer']
            if c not in class_counts:
                class_counts[c] = 0
            class_counts[c] += 1
        
        for key in class_counts:
            class_counts[key] = round(class_counts[key] / len(self.entries), 2)
        return class_counts

    def __str__(self):
        return f"LVIS dataset with {len(self.entries)} questions"


    def __len__(self):
        return len(self.entries)


    def __getitem__(self, index):
        entry = self.entries[index]

        return entry