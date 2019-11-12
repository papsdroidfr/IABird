#!/usr/bin/env python3
#############################################################################
# Filename    : IABird.py
# Description : Flappy BIrd qui apprend à jouer tout seul avec un réseau de neurones
# Author      : papsdroid.fr
# modification: 2019/11/11
########################################################################


import tkinter as tk
import random as rd
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


#terrain de jeu
class Terrain:
    def __init__(self, canv, height, width):
        self.canv=canv      # canvas sur lequel le terrain est déssiné
        self.h=height       # hauteur
        self.w=width        # largeur
        self.animation = 4  # pas de défilement en nb de pixel des barières entre chaque cycle
        self.barrieres = [] # liste des barrières actives
        self.nuages = []    # liste de nuages
        self.bird=None      # oiseau
        self.delais_prochaine_bar=0     #nb de cycle avant apparition d'un prochaine barière
        self.delais_prochain_nuage=0    #nb de cycle avant apparition d'un prochain nuage
        self.img_herbe_png=Image.open('herbe.png')
        self.img_herbe=ImageTk.PhotoImage(self.img_herbe_png)     
        #self.add_bird('humain', posX=90, comportement=-1)
        self.add_bird('IA_1', posX=90, comportement=1)
        self.commandeVolSouris = False     #False=chutte, True=vol :commande vol à la souris pour un contrôle par humain
             
    def initialise(self):
        self.barrieres = []
        self.delais_prochaine_bar=0     #nb de cycle avant apparition d'un prochaine barière
        self.bird.initialise()     
        
    def add_barriere(self):
        self.barrieres.append(Barriere(self))
        
    def add_nuage(self):
        self.nuages.append(Nuage(self))
    
    def add_bird(self, nom, posX, comportement):
        self.bird = Bird(self, nom, posX, comportement)
        
    def dessine(self):
        #supprime tous les objets du canvas
        self.canv.delete(tk.ALL)
        #dessin herbe
        self.canv.create_image(0,self.h-self.img_herbe.height(),anchor=tk.NW, image=self.img_herbe)
        self.canv.create_image(self.w,self.h-self.img_herbe.height(),anchor=tk.NE, image=self.img_herbe)
        #dessin des nuages en arrières plan
        for n in self.nuages:
            n.dessine()
        #dessine l'oiseau
        self.bird.dessine()
        #dessin des barières
        for b in self.barrieres:
            b.dessine()
            
        
#obstacle de type barrière (animé)
class Barriere:
    def __init__(self, terrain):
        self.terrain = terrain
        self.visible = True  # true=visible, False=invisible
        self.taille=rd.randint(terrain.bird.taille//2,terrain.bird.taille*3)       # largeur de la barrière
        self.coin=3          # rayon des cercles d'identification sur les coins
        self.ouverture = rd.randint(terrain.bird.taille*3, terrain.bird.taille*4 ) # hauteur de l'ouverture en nb de pixels
        self.x = terrain.w   # x,y = position haut-gauche de la barrière sur le terrain
        self.y = 0
        self.milieu = terrain.h//2  + rd.randint(-terrain.h//4,terrain.h//4) #milieu de l'ouverture
        self.coins_gche = False  # affichage des coins gauche (haut et bas)
        self.coins_dte = False   # affichage des coins droite (haut et bas)
        self.identifie_coins()
        
    #identifie la position dess 4 coins de l'ouverture hau_gauche, haut_doit, bas_gauche, bas_droit
    def identifie_coins(self):
        self.x_gche = self.x
        self.x_dte = self.x + self.taille
        self.y_haut = self.milieu - self.ouverture//2
        self.y_bas = self.milieu + self.ouverture//2
        
    def dessine(self):
        self.identifie_coins()
        #haut de la barrière
        self.terrain.canv.create_rectangle(self.x, self.y , 
                                           self.x+self.taille, self.milieu-self.ouverture//2,
                                           fill='black', outline='yellow', width=2, tags='b')
        #bas de la barrière
        self.terrain.canv.create_rectangle(self.x, self.milieu+self.ouverture//2, 
                                           self.x+self.taille, self.terrain.h,
                                           fill='black', outline='yellow', width=2, tags='b')
        
        #affichage coins gche et bas
        if self.coins_gche:
            self.terrain.canv.create_oval(self.x_gche-self.coin, self.y_haut-self.coin,
                                          self.x_gche+self.coin, self.y_haut+self.coin, fill='red')
            self.terrain.canv.create_oval(self.x_gche-self.coin, self.y_bas-self.coin,
                                          self.x_gche+self.coin, self.y_bas+self.coin, fill='red')
        if self.coins_dte:
            self.terrain.canv.create_oval(self.x_dte-self.coin, self.y_haut-self.coin,
                                          self.x_dte+self.coin, self.y_haut+self.coin, fill='red')
        
            self.terrain.canv.create_oval(self.x_dte-self.coin, self.y_bas-self.coin,
                                          self.x_dte+self.coin, self.y_bas+self.coin, fill='red')

#nuage (animé)
class Nuage:
    def __init__(self, terrain):
        self.terrain = terrain
        self.img_png=Image.open('nuage1.png')
        self.img=ImageTk.PhotoImage(self.img_png)
        self.x = terrain.w + int(self.img.width()/2)         #position du nuage
        self.y = rd.randint(int(terrain.h*10/100), int(terrain.h*40/100)) 
        self.visible=True
        
    def dessine(self):
        self.terrain.canv.create_image(self.x,self.y,image=self.img)

#Reinforcement learning: neural Q-network with policy gradient
#  IA -1= human control
#  IA 0 = basic, not learning (default value)
#  IA 1 = DQN with no knowledge (SCORE = REWARDS)
#  IA 2 = DQN with Knowledge injected in rewards
class DQN_learning():
    def __init__(self, bird):
        self.IA = bird.comportement     # -1, 0, 1 ,2 : see above comment
        self.bird = bird
        self.n_games_per_update = 10     # number of games done before calculating mean rewards
        self.episode_number = 1         # 1 episode = 1 game completed
        self.current_iteration = 0      # 1 iteration = n_games_per_update" games completed 
        self.reward=0                   # total rewards of the previous episode
        self.score=0                    # total score of the previous episode
        self.mean_reward=0              # mean rewards of previous iteration
        self.mean_score=0               # mean score of the previous iteration
        self.var_score=0                # variance score of the previous iteration
        self.current_rewards=[]         # all rewards from the current episode
        self.current_score=[]           # all records from the current episode
        self.all_rewards=[]             # sequences of rewards for each episode
        self.all_score=[]               # sequences of records for each episode
        self.mean_scores=[]             # scores from the current episode to calculate mean score
        self.data_score=[]              # historical of mean records for all episode
        
        if self.IA>0:
            print('Inializing Neural Network, may take a while ....')
            tf.reset_default_graph()    # to compute gradients, graph needs to be reset to default
            self.graph = tf.Graph()
            self.n_inputs = len(self.bird.obs)  # inputs = observation
            self.n_hidden1 = 4           # hidden layer of the neural network
            #self.n_hidden2 = 4           # hidden layer of the neural network
            self.n_outputs = 1           # output = probalility going left (action 0: accelerate left, acction 1: accelerate rigth)
            #self.initializer = tf.variance_scaling_initializer()
            self.initializer = tf.contrib.layers.variance_scaling_initializer()
            self.learning_rate = 0.01   # optimizer learning rate   
            self.discount_rate = 0.99   # aslo called "gamma" factor
            
            #neural network configuration
            self.X = tf.placeholder(tf.float32, shape=[None, self.n_inputs])
            self.hidden1 = tf.layers.dense(self.X, self.n_hidden1, activation=tf.nn.elu, kernel_initializer=self.initializer)
            #self.hidden2 = tf.layers.dense(self.hidden1, self.n_hidden2, activation=tf.nn.elu, kernel_initializer=self.initializer)
            self.logits = tf.layers.dense(self.hidden1, self.n_outputs)
            self.outputs = tf.nn.sigmoid(self.logits)  # probability of action 0 (VOL), 1-outputs = probability of inverse action (CHUTTE)
            self.p_left_and_right = tf.concat(axis=1, values=[self.outputs, 1 - self.outputs])
            self.action = tf.multinomial(tf.log(self.p_left_and_right), num_samples=1) #random action 0 or 1, based on output probalibility
        
            #gradient descent policy
            self.y = 1.0 - tf.to_float(self.action) #target probability: must be 1.0 if chosen action is 1 (VOL) and 0.0 if chosen action is 0 (CHUTTE)
            self.cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y, logits=self.logits)
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
            self.grads_and_vars = self.optimizer.compute_gradients(self.cross_entropy)
            self.gradients = [grad for grad, variable in self.grads_and_vars]
            self.gradient_placeholders = []
            self.grads_and_vars_feed = []
            for grad, variable in self.grads_and_vars:
                gradient_placeholder = tf.placeholder(tf.float32)
                self.gradient_placeholders.append(gradient_placeholder)
                self.grads_and_vars_feed.append((gradient_placeholder, variable))
            self.training_op = self.optimizer.apply_gradients(self.grads_and_vars_feed)
            self.init = tf.global_variables_initializer()
            
            #TF session initialisation
            self.init = tf.initialize_all_variables()
            self.sess = tf.Session()
            self.sess.run(self.init)
            self.all_gradients=[]     # gradient saved at each step of each episode    
            self.current_gradients=[] # all grandients from the current episode
            print('Complete!')

    #compute total rewards, given rewards and discount_rate 
    def discount_rewards(self, rewards):
        discounted_rewards = np.empty(len(rewards))
        cumulative_rewards=0
        for step in reversed(range(len(rewards))):
            cumulative_rewards = rewards[step] + cumulative_rewards * self.discount_rate
            discounted_rewards[step] = cumulative_rewards
        return discounted_rewards

    #nomalize rewards accros multiple episodes
    #return normalized score for each action in each episode
    #all_rewards [a,b,c], [d,e] ....
    #return: [a',b',c'], [d',e'] normalized scores
    def discount_and_normalize_rewards(self):
        all_discount_rewards = [self.discount_rewards(rewards) for rewards in self.all_rewards]
        flat_rewards = np.concatenate(all_discount_rewards)
        reward_mean = flat_rewards.mean()
        reward_std = flat_rewards.std()
        return [(discounted_rewards - reward_mean)/reward_std for discounted_rewards in all_discount_rewards]
    
    # Run policy either by learning (IA=True) or with basic policy (IA=False)
    def learn(self, obs):            
        #takes decision (up or not), based on policy (either IA or basic)
        if self.IA<0: #human control
            action = 1 if self.bird.terrain.commandeVolSouris else 0
        elif self.IA>0: #neural network decision
            action_val, gradients_val = self.sess.run([self.action, self.gradients], feed_dict={self.X: obs.reshape(1, self.n_inputs)})
            self.current_gradients.append(gradients_val)
            action=action_val[0][0]
        else: #basic policy: vol si bird en dessous du milieu de l'ouverture
            action=1
            if self.bird.bar is not None:
                action = 1 if self.bird.bar.y_bas - (self.bird.y + self.bird.taille//2)  < self.bird.taille//3 else 0  
            
       
        #new observation, folowing action
        new_obs, reward, done = self.bird.bouge(action)
        self.current_score.append(reward) #score = reward from the env
        #par defaut reward = 1, ajout de malus si trop éloigné du bord bas
        if self.IA==2:
            reward -=  abs(new_obs[0])/5  # malus si trop loin des bords
        
        self.current_rewards.append(reward)
        self.score = np.sum(self.current_score)
        
        if done: #game completed
            self.all_rewards.append(self.current_rewards)
            self.all_score.append(self.current_score)
            self.reward = np.sum(self.current_rewards)
            #self.score = np.sum(self.current_score)
            self.mean_scores.append(self.score)
            self.current_rewards=[]   # all rewards from the current episode
            self.current_score=[]     # all score from the current episode
            if self.IA>0:
                self.all_gradients.append(self.current_gradients)
                self.current_gradients=[] # all grandients from the current episode
            #reset the environnement
            self.bird.terrain.initialise()
            
            #update params & learn after n_games_per_update games
            if (self.episode_number % self.n_games_per_update) == 0:
                #rewards mean
                self.mean_score = np.mean(self.mean_scores)
                self.var_score = np.std(self.mean_scores)
                self.data_score.append(self.mean_score)
                print('IA{}-Itération {}*{} games - mean score {} , variance {}'.format(
                        self.IA, self.current_iteration, self.n_games_per_update, 
                        self.mean_score, self.var_score,                 
                        ))
                self.mean_scores= []
                self.current_iteration += 1
                if self.IA>0:
                    self.all_rewards = self.discount_and_normalize_rewards()
                    feed_dict = {}
                    for var_index, grad_placeholder in enumerate(self.gradient_placeholders):
                        #multiply the gradient by the action score, and compute the mean
                        mean_gradients = np.mean(
                            [reward * self.all_gradients[game_index][step][var_index]
                            for game_index, rewards in enumerate(self.all_rewards)
                            for step, reward in enumerate(rewards)],
                            axis=0)
                        feed_dict[grad_placeholder] = mean_gradients
                    self.sess.run(self.training_op, feed_dict=feed_dict)
                    self.all_gradients=[]  # gradient saved at each step of each episode
                self.all_rewards=[]        # sequences of rewards for each episode
                self.all_score=[]          # sequences of score for each episode
            self.episode_number += 1  
            
        return action, new_obs, reward, done           
    
#oiseau, 
class Bird():
    def __init__(self, terrain, nom, posX, comportement=0):
        self.img_png = Image.open('bird_50.png')
        self.img_png_hurted = Image.open('bird_50_hurted.png')
        self.img = ImageTk.PhotoImage(self.img_png)
        self.img_hurted = ImageTk.PhotoImage(self.img_png_hurted)
        self.taille = self.img.height()
        self.terrain = terrain
        self.nom = nom
        self.comportement = comportement     # -1=controlé par humain, 0=automatique, 1=IA
        self.x = posX                        # position sur le terrain (x,y) = coordonnées du milieu de l'image de l'agent
        self.pas = 5                         # déplacement vertical entre chaque cycle.
        self.delta_y_max = 4*self.pas        # pour limiter l'effet de l'accélération de la chutte
        self.delta_y_vol = -3*self.pas       # saut vertical maxi lors d'un vol
        self.accel_y  = 2                    # accélération chutte entre chaque cycle
        self.scoreMAX = 20                   # perfection quand scoreMAX atteint (score remis à zéro)
        self.initialise()
        self.learning = DQN_learning(self)   # learning capabilities
           
    def initialise(self):
        self.y = self.terrain.h//2
        self.action = 0               # 0=chutte, 1=vol      
        self.delta_y  = self.pas             # delta_y > 0 = chutte (0 en haut)
        self.hurted = False                  # True si collision avec un obstacle, False sinon
        self.franchissement = False          # True si une barrière est en cours de franchissement
        self.score = 0                       # nb de barrieres franchies entre 2 echecs (remis à zéro à chaque échec)
        self.current_bar = True              # False si barrière en cours est heurtée, True sinon: remis à True à chaque barrière franchie
        self.compteBarriere = True           # False si comptage barrière franchie non encore effectué
        self.bar = None                      # barrière en cours à franchir
        self.set_obs()                       # observation of current iteration 

    def dessine(self):
        if self.hurted:
            self.terrain.canv.create_image(self.x,self.y,image=self.img_hurted, anchor='center')
        else:
            self.terrain.canv.create_image(self.x,self.y,image=self.img,anchor='center')          
        #score et nom
        self.terrain.canv.create_text(self.x, self.y - int(self.img.height()/2)-5,text=self.nom, fill='yellow')    
        self.terrain.canv.create_text(self.x, self.y + int(self.img.height()/2)+5,
                                      text="reward:{}".format(self.learning.score),fill='white')
        self.terrain.canv.create_text(self.x, self.y + int(self.img.height()/2)+20,
                                      text="bar:{}".format(self.score),fill='white')
        #observations
        if self.bar:
            self.terrain.canv.create_line(self.x+self.taille//2, self.y-self.taille//2,self.bar.x_gche, self.bar.y_haut, fill='red', dash=(3,2))
            self.terrain.canv.create_line(self.x+self.taille//2, self.y+self.taille//2,self.bar.x_gche, self.bar.y_bas, fill='red', dash=(3,2))
            if self.franchissement:
                self.terrain.canv.create_line(self.x+self.taille//2, self.y-self.taille//2,self.bar.x_dte, self.bar.y_haut, fill='red', dash=(3,2))
                self.terrain.canv.create_line(self.x+self.taille//2, self.y+self.taille//2,self.bar.x_dte, self.bar.y_bas, fill='red', dash=(3,2))   
        
    #observation de l'environnement lors d'un cycle pour prise de décision
    def set_obs(self):
        if(self.bar is not None):
            self.bar.coins_gche = True
            self.bar.coins_dte = False
            if self.franchissement:
                self.bar.coins_dte = True
            self.obs[0] = (self.bar.x_gche-self.x-self.taille//2)/self.terrain.w  if not(self.franchissement) else 0 # position relative du bird par rapport à la barrière 
            self.obs[1] = (self.bar.y_bas-self.y-self.taille//2)/self.bar.ouverture  #dist sous plafond bas relative à la taille de l'ouverture
            #self.obs[2] = self.delta_y/self.delta_y_max   #acceleration en cours            
            #print(self.obs)
        else: 
            self.obs=np.array([1,1], dtype='f') 

    #fait bouger l'oiseau selon l'action choisie (1=vol, 0=chutte) + test de collision
    def bouge(self, action):
        reward, done, self.action = 0, False, action
        #nouvelle position et vitesse du bird en fonction de l'action
        if (self.action==1):
            self.delta_y = self.delta_y_vol     # accélération verticale de vol
        else:
            self.delta_y += self.accel_y        # accélération verticale de chutte
        #if (self.delta_y > self.delta_y_max):   # limitation de l'accélération
        #    self.delta_y = self.delta_y_max
        y = self.y + self.delta_y
        if (y>self.taille//2 and y+self.taille//2<self.terrain.h): #bord haut et bas du terrain
            self.y = y
        else: #bords terrain heurté
            #print('Score: {}, Bords heurtés!'.format(self.score))
            self.delta_y=0  
            self.score = 0
            done = True
        
        #test de collision et calcul du reward: +1 si pas de collsision
        self.hurted = self.test_collision() 
        if self.hurted:
            done=True
        else:
            reward=1
        
        if self.franchissement: # franchissement en cours
            if self.hurted:     # la barriere a été heurtée
                self.current_bar = False
        #mise à jour du score si barrière totalement franchie + maj reward si barrière franchie avec succès
        else: #franchissement terminé
            if not(self.compteBarriere):
                if self.current_bar: #barrière franchie sans aucune collision
                    self.score +=1
                else: #barrière franchie mais heurtée
                    self.score = 0
                if self.score==self.scoreMAX: #score max atteint
                    #print('Score: {}, Bravo MAX atteint!'.format(self.score))
                    self.score=0
                    done = True
                self.compteBarriere=True
                self.current_bar = True
        
        self.set_obs() #maj observation
        return self.obs, reward, done
     
    #test de collision
    def test_collision(self):
        collision = False #True si bird a percuté un obstacle, False sinon
        if len(self.terrain.barrieres)>0:
            i, b = 0, self.terrain.barrieres[0]
            #cherche la bariere b devant le bird
            while ( self.x-self.taille//2 > b.x+b.taille): #tant que le bord gche du bird est <= au bord droit d'une barrière
                i += 1
                b = self.terrain.barrieres[i]      
            self.bar = b
            
            if ( (self.x+self.taille//2) >= b.x and #le bord droit du bird est déjà devant le bord gauche de la barrière
                 (self.x-self.taille//2) <= b.x + b.taille): #le bord gauche du bird n'a pas dépassé l'extrémité de la barrière
                #le bird franchi la barrière: test de collision des bords haut et bas du bird avec l'ouveture
                self.franchissement = True      #franchissement de barrière en cours
                if self.compteBarriere:         
                    self.compteBarriere = False #comptage barrière en cours pour score
                if ( ((self.y-self.taille//2) < (b.milieu-b.ouverture//2)) or #test collision bord haut du bird 
                     ((self.y+self.taille//2) > (b.milieu+b.ouverture//2)) ): #test collision bord bas du bird
                    collision = True            
            else:
                self.franchissement = False
        return collision
 
        
#interface graphique simple sous Tkinter
class Application:
    def __init__(self,height=600,width=800,delais=10): 
        self.height=height  #hauteur
        self.width=width    #largeur
        self.delais=delais  #délais de rafraichissement de l'animation
        self.flag_play = 0  #1=animation en cours
        self.nb_cycles = 0  # nb de cycles
        self.render = True      #True = animation graphique activée (apprentissage neuronal bpc plus lent)
        self.delais_plot = 50   #graphe quand délais d'itération écoulé
        self.graphe = True      #True si graphe cycle en cours déjà affiché
        
        #interface graphique    
        self.fen1 = tk.Tk()
        self.fen1.title('P@psDroid - IA Bird')
        self.fen1.resizable(width='No', height='No')
        self.can1 = tk.Canvas(self.fen1, width=width, height=height, bg ='blue')
        self.can1.pack(side=tk.TOP, padx=5, pady=5)
        #boutons GO et STOP, RENDER et QUIT
        self.b1 = tk.Button(self.fen1, text ='Go!', command =self.go)
        self.b2 = tk.Button(self.fen1, text ='Stop', command =self.stop)
        self.b3 = tk.Button(self.fen1, text ='Render', command =self.rendering)
        self.b4 = tk.Button(self.fen1, text ='Quit', command =self.quitter)
        self.b1.pack(side =tk.LEFT, padx =3, pady =3)
        self.b2.pack(side =tk.LEFT, padx =3, pady =3)
        self.b3.pack(side =tk.LEFT, padx =3, pady =3)
        self.b4.pack(side =tk.LEFT, padx =3, pady =3)
        #label nbcycles
        self.labcycles = tk.Label(self.fen1)
        self.labcycles.configure(text="Cycles:")
        self.labcycles.pack(side=tk.LEFT,padx =3, pady =3)
        self.labnbcycles = tk.Label(self.fen1)
        self.labnbcycles.configure(text="0")
        self.labnbcycles.pack(side=tk.LEFT,padx =3, pady =3)
        #terrain principal
        self.terrain = Terrain(self.can1, height=height, width=width)
        self.terrain.dessine()
        #self.fen1.bind("<KeyPress-space>",self.action_vole)
        #self.fen1.bind("<KeyRelease-space>",self.action_chutte)
        self.fen1.bind('<Button-1>',self.action_vole)
        self.fen1.bind('<ButtonRelease-1>',self.action_chutte)
        #boucle principale
        self.fen1.mainloop()

    def action_vole(self, event):
        self.terrain.commandeVolSouris = True
    
    def action_chutte(self, event):
        self.terrain.commandeVolSouris = False
        
    def go(self):
        if self.flag_play==0:
            print('Animation started')
            self.flag_play=1
            self.play()

    def plotEvol(self):
        plt.figure(figsize=(12,6))
        plt.title("Evolution des Rewards par itération")
        plt.xlabel('itérations')
        plt.ylabel('Rewards')
        plt.grid(True,linestyle='dashed')
        plt.plot(self.terrain.bird.learning.data_score, label=self.terrain.bird.nom)
        plt.legend()
        plt.show()
        
    def stop(self):
        self.flag_play = 0
        print('Animation stopped')
        self.plotEvol()


    def rendering(self):
        self.render = not(self.render) #inverse l'option de dessin
        if self.render:
            print('rendering ON (low learning)')
        else:
            print('rendering OFF (fast learing)')
    
    def quitter(self):
        print('Bye')
        #closing sessions
        if self.terrain.bird.comportement>0:
            self.terrain.bird.learning.sess.close()
        self.fen1.destroy()
        self.fen1.quit()         

    def play(self):
        if (self.flag_play>0):  
            
            #animation des nuages
            for n in self.terrain.nuages:
                n.x -= self.terrain.animation//2    #ils vont 2 fois moins vite que les barrirèes
                if(n.x < -n.img.width()):
                    n.visible = False
            self.terrain.nuages=[el for el in self.terrain.nuages if el.visible]    #suppression des nuages devenus invisibles
            if self.terrain.delais_prochain_nuage <= 0: #test apparition nouveau nuage
                self.terrain.add_nuage()
                self.terrain.delais_prochain_nuage = rd.randint(int(300/self.terrain.animation),int(800/self.terrain.animation))
            
            #animation des barrières
            for b in self.terrain.barrieres:
                b.x -= self.terrain.animation
                if (b.x < -b.taille):
                    b.visible = False             
            if self.terrain.delais_prochaine_bar <= 0:  #test apparition nouvelle barrière
                self.terrain.add_barriere()
                #self.terrain.delais_prochaine_bar = rd.randint(200//self.terrain.animation,400//self.terrain.animation)
                self.terrain.delais_prochaine_bar = 70
            self.terrain.barrieres=[el for el in self.terrain.barrieres if el.visible]  #suppression des barières devenues invisibles
            
            # animation bird
            self.terrain.bird.set_obs()
            action, self.terrain.bird.obs, reward, done = self.terrain.bird.learning.learn(self.terrain.bird.obs)           
            
            # mise à jour des cycles    
            self.nb_cycles += 1         
            self.labnbcycles.configure(text=self.nb_cycles) #affichage du nb de cycles
            self.terrain.delais_prochaine_bar-=1
            self.terrain.delais_prochain_nuage-=1

            #animation graphique    
            if self.render:
                self.terrain.dessine()      # dessin du terrain
                self.fen1.after(self.delais,self.play)
            else:
                self.fen1.after(0,self.play)         

#interface graphique    
appl = Application() #instancation d'un objet Application


