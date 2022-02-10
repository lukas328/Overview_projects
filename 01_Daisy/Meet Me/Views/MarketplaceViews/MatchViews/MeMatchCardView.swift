//
//  MeMatchView.swift
//  Meet Me
//
//  Created by Philipp Hemkemeyer on 21.02.21.
//

import SwiftUI
import URLImage

struct MeMatchCardView: View {
    
    @ObservedObject var meMatchCardVM: MeMatchCardViewModel = MeMatchCardViewModel()
    // binding vars
    @Binding var userAccepted: Bool
    @Binding var users: [UserModel]
    @Binding var event: EventModel
    @Binding var showMeMatchMainView: Bool
    var user: UserModel
    var userNumber: Int
    
    init(userChosen: Binding<Bool>, users: Binding<[UserModel]>, event: Binding<EventModel>, user: UserModel, userNumber: Int, showMeMatchMainView: Binding<Bool>) {
        self._userAccepted = userChosen
        self._users = users
        self._event = event
        self.user = user
        self.userNumber = userNumber
        self._showMeMatchMainView = showMeMatchMainView
    }
    
    // internal vars
    var screenWidth: CGFloat? = 340
    
    @State var translation: CGSize = .zero
    @State var degrees: Double = 0
    
    @State var userDenied: Bool = false
    
    var body: some View {
        
        let dragGesture = DragGesture()
            .onChanged{ (value) in
                translation = value.translation
                
                // state is change to 2 if the drag is to the right and -2 if to the left
                if value.translation.width > 0 {
                    degrees = Double(translation.width / 10)
                    if value.translation.width > 80 {
                        // secure that each user is only accepted once
                        if !userAccepted {
                            
                            userWasAccepted()
                        }
                        
                    }
                } else {
                    degrees = (Double((translation.width / 10)))
                    if value.translation.width < -80 {
                        // secure that each user is only denied once
                        if !userDenied {
                            userWasDenied()
                        }
                        
                    }
                }
            }
            .onEnded { value in
                // go back to the normal view, when not swiped over critical point
                if !userDenied || !userAccepted {
                    translation = .zero
                    degrees = 0
                }
            }
        
        ZStack {
            //user.UserPhotos.url
            URLImage(url: URL(string: user.userPhotos[0] ?? stockUrlString)!  ) { image in
                image.resizable()
                    .aspectRatio(contentMode: .fill)
                    .frame(width: screenWidth, height: 620, alignment: .center)
            }
            
            
            
            VStack {
                
                // push all details down to show more of the picture
                Spacer()
                
                // MARK: Text Content
                VStack {
                    
                    // MARK: Profile information
                    HStack {
                        Text(user.name)
                            .font(.title)
                            .foregroundColor(.accentColor)
                        Text(String(dateToAge(date: user.birthdayDate)))
                            .font(.body)
                    }
                    .padding(.top, 5)
                    .padding(.bottom, -5)
                    .padding(.horizontal, 12)
                    .frame(width: screenWidth, alignment: .leading)
                    
                    Divider()
                    
                    // MARK: - Button area
                    HStack {
                        
                        // MARK: Deny Match - Button
                        ZStack {
                            Image(systemName: "xmark.circle")
                                .font(.title)
                                .foregroundColor(.red)
                            
                        }
                        .frame(width: ((screenWidth! - 40) * 2/3))
                        .padding(4)
                        .background(Color.red.opacity(0.3))
                        .onTapGesture {
                            // update database - user not
                            userWasDenied()
                        }
                        
                        
                        // MARK: Accept Match - Button
                        ZStack {
                            Image(systemName: "checkmark.circle")
                                .font(.title)
                                .foregroundColor(.green)
                        }
                        .frame(width: (screenWidth! - 40) * 1/3)
                        .padding(4)
                        .background(Color.green.opacity(0.3))
                        .onTapGesture {
                            // user was taken
                            
                            userWasAccepted()
                        }
                        
                        
                    }
                    .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
                    .padding(.bottom, 12)
                    
                    
                    
                }
                .background(BlurView(style: .systemThinMaterial))
                
            }
            
            
        }
        .frame(width: screenWidth, height: 620, alignment: .center)
        .modifier(FrozenWindowModifier())
        .offset(x: userDenied ? -400 : 0)
        .offset(x: userAccepted ? 400 : 0)
        .offset(x: translation.width, y: 0)
        
        .rotationEffect(.degrees(degrees))
        
        .gesture(dragGesture)
        .animation(Animation.interactiveSpring().speed(0.2))
    }
    
    // MARK: - functions
    
    // MARK: function which gets called after user accepted the profile
    func userWasAccepted() {
        meMatchCardVM.addMatch(eventModel: event, userModel: user)
        userAccepted = true
        DispatchQueue.main.asyncAfter(wallDeadline: .now() + 0.1) {
            showMeMatchMainView = false
        }
    }
    
    // MARK: function which gets called after user denied the profile
    func userWasDenied() {
        userDenied = true
        meMatchCardVM.deleteLikedUser(eventModel: event, userModel: user)
        // if the last user (which is the first in the array) is denied the view gets canceled
        if self.users.first!.userId == user.userId {
            meMatchCardVM.setLikedUserToFalse(eventId: event.eventId)
            DispatchQueue.main.asyncAfter(wallDeadline: .now() + 0.1) {
                showMeMatchMainView = false
            }
        }
    }
    
}

struct MeMatchCardView_Previews: PreviewProvider {
    static var previews: some View {
        MeMatchCardView(userChosen: .constant(false), users: .constant([stockUser, stockUser]), event: .constant(stockEvent), user: stockUser , userNumber: 1, showMeMatchMainView: .constant(false))
    }
}
