package com.example.neural_network_backend.controller;

import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.Calendar;
import java.util.List;
import java.util.Optional;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.messaging.handler.annotation.MessageMapping;
import org.springframework.messaging.handler.annotation.Payload;
import org.springframework.messaging.handler.annotation.SendTo;
import org.springframework.messaging.simp.SimpMessagingTemplate;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.CrossOrigin;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import com.example.neural_network_backend.model.Chat;
import com.example.neural_network_backend.model.MessagesDTO;
import com.example.neural_network_backend.repository.ChatRepository;
import com.example.neural_network_backend.repository.MessageRepository;



// @Controller
public class ChatController {
    
    @Autowired
    private SimpMessagingTemplate simpMessagingTemplate;

    @Autowired 
    private MessageRepository messageRepo;

    @Autowired 
    private ChatRepository chatRepo;

    @MessageMapping("/message")
    @SendTo("/chatroom/public")
    private MessagesDTO receivePublicMessage(@Payload MessagesDTO message){
       return message; 
    }

    @MessageMapping("/private-message")
    public List<MessagesDTO> receivePrivateMessage(@Payload MessagesDTO message){
        
        
        // so now we need to add the 
        
        // we need to create a room for the reciever and the sender 
        String a = message.getSenderName();
        String b = message.getReceiverName();
        String c = "";



//      An int value: 0 if the string is equal to the other string.
//      < 0 if the string is lexicographically less than the other string
//      > 0 if the string is lexicographically greater than the other string (more characters)

        
        if(a.compareTo(b) == 0){
            c = a + b;
        }

        if(a.compareTo(b) < 0){
            c = b + a;
        }

        if(a.compareTo(b) > 0){
            c = a + b;
        }

     

        Optional<Chat> room = chatRepo.findByRoom(c);

        if(room.isPresent()){
            message.setChatRoomId(room.get().getRoomId());
            messageRepo.findByChatRoomId(room.get().getRoomId());
            
            System.out.println("----------xWeelll: " + c);
            
            message.setDate(new SimpleDateFormat("yyyy/MM/dd HH:mm:ss").format(Calendar.getInstance().getTime()));
            messageRepo.save(message);
            List<MessagesDTO> m = messageRepo.findByChatRoomId(room.get().getRoomId());
            System.out.println("----------xWeelll222: " + m.size());
            simpMessagingTemplate.convertAndSendToUser(message.getReceiverName(), "/private" , message);
            return messageRepo.findByChatRoomId(room.get().getRoomId());
        }else {
            Chat newChat = new Chat();
            newChat.setRoom(c);
            chatRepo.save(newChat);
            System.out.println("----------xHEloo: " +  newChat.getRoomId());
            message.setChatRoomId(newChat.getRoomId());
            message.setDate(new SimpleDateFormat("yyyy/MM/dd HH:mm:ss").format(Calendar.getInstance().getTime()));
            messageRepo.save(message);
            simpMessagingTemplate.convertAndSendToUser(message.getReceiverName(), "/private" , message);

            return messageRepo.findByChatRoomId(newChat.getRoomId());
            
        }
        
        
        // if they dont have a chatroom than create a new chatRoom and 
        // so now we need to store the message into the data base 
        
        // now we need to create a chatRoom that holds all of the items 
       
    }

   
}

