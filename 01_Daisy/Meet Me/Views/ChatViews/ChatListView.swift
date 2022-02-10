//
//  ChatListView.swift
//  Meet Me
//
//  Created by Philipp Hemkemeyer on 27.02.21.
//

import SwiftUI

struct ChatListView: View {
    
    @StateObject var chatListVM: ChatListViewModel = ChatListViewModel()
    
    
    @State var chatTapped: Bool = false
    @State var matchTapped: AllMatchInformationModel = AllMatchInformationModel(chatId: "", unReadMessage: false, user: stockUser, event: stockEvent)
    @State var matchLongPressed: AllMatchInformationModel = AllMatchInformationModel(chatId: "", unReadMessage: false, user: stockUser, event: stockEvent)
    
    var body: some View {
        
        
        NavigationView {
            VStack {
                
                HeaderView(text1: "Chat ", text2: " Area", highlightText: "ME")
                
                // chatListView is not created when there is no match
                if !chatListVM.matches.isEmpty {
                    
                    ScrollView {
                        VStack {
                            
                            // space to push the view a bit more down
                            Color.clear.frame(height: 15)
                            
                            // listing chats inside view
                            ForEach(chatListVM.matches.indices, id: \.self) { matchNumber in
                                VStack(spacing: 15.0) {
                                    
                                    NavigationLink(
                                        destination: MessagesView(match: $matchTapped),
                                        isActive: $chatTapped
                                    )
                                    {
                                        ChatListRowView(match: $chatListVM.matches[matchNumber], chatTapped: $chatTapped, matchTapped: $matchTapped, matchLongPressed: $matchLongPressed)
                                            
                                        
                                    }
                                    // menu to dissolve match or delete and dissolve match
                                    .contextMenu {
                                        
                                        // Button to only dissolve this match the event goes back to the pool
                                        Button {
                                            chatListVM.deleteMatchAndBackToPool(match: chatListVM.matches[matchNumber], index: matchNumber)
                                        } label: {
                                            Label("Dissolve Match", systemImage: "person.crop.circle.badge.minus")
                                        }
                                        
                                        // Button to dissolve the match and delete the whole event, only available when the current user is the creator
                                        if chatListVM.checkIfUserIsCreatorFromEvent(creatorId: chatListVM.matches[matchNumber].event.userId) {
                                            Button {
                                                chatListVM.deleteMatchAndEventCompletely(match: chatListVM.matches[matchNumber], index: matchNumber)
                                            } label: {
                                                Label("Dissolve Match and delete Event", systemImage: "minus.circle")
                                            }
                                        }
                                        
                                        // Button to report a user and dissolve the match
                                        Button {
                                            chatListVM.reportUser(match: chatListVM.matches[matchNumber])
                                        } label: {
                                            Label("Report user and dissolve Match", systemImage: "exclamationmark.circle")
                                        }
                                        
                                    }
                                    
                                }
                                
                            }
                        }
                        
                        
                        
                    }
                } else {
                    
                    Spacer()
                    
                    Text(chatListVM.messageIfNoMatches)
                        .padding()
                    
                    Spacer()
                }
            }
            .frame(maxWidth: .infinity)
            .background(
                Image("background")
                    .resizable()
                    .aspectRatio(contentMode: .fill)
            )
            
            .navigationBarHidden(true)
            
        }
        .onAppear {
//            chatListVM.activateListner().catch { error in
//                print(error)
//            }
            chatListVM.getMatches()
        }
        
    }
    
}

struct ChatListView_Previews: PreviewProvider {
    static var previews: some View {
        ChatListView()
    }
}
