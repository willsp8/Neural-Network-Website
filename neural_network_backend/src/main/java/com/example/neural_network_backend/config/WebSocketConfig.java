package com.example.neural_network_backend.config;

import org.springframework.context.annotation.Configuration;
import org.springframework.web.socket.config.annotation.EnableWebSocketMessageBroker;
import org.springframework.messaging.simp.config.MessageBrokerRegistry;
import org.springframework.web.socket.config.annotation.StompEndpointRegistry;
import org.springframework.web.socket.config.annotation.WebSocketMessageBrokerConfigurer;


@Configuration
@EnableWebSocketMessageBroker
public class WebSocketConfig implements WebSocketMessageBrokerConfigurer{

    @Override
    public void configureMessageBroker(MessageBrokerRegistry registry){

        // this sets up topic prefixes 
        registry.setApplicationDestinationPrefixes("/app");
        registry.enableSimpleBroker("/chatroom", "/private/message","/user", "/private-game", "/game", "/topic", "/hello", "/helloo", "/news", "/move");
        registry.setUserDestinationPrefix("/user");

    }


    @Override
    public void registerStompEndpoints(StompEndpointRegistry registry){

        // this serves as the starting path for all of our websocket calls 
        registry.addEndpoint("/ws")
        .setAllowedOriginPatterns("*");
    }
    
    
}

