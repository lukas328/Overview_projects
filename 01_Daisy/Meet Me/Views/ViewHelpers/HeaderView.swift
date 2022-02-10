//
//  HeaderView.swift
//  Meet Me
//
//  Created by Philipp Hemkemeyer on 13.03.21.
//

import SwiftUI
import PromiseKit

struct HeaderView: View {
    
    var userManger: FirestoreManagerUserTest = FirestoreManagerUserTest()
    @State var user: UserModel = stockUser
    
    var text1: String = "That's "
    var text2: String = "!"
    var highlightText: String = "ME"
    
    var body: some View {
        
        ZStack {
            
            
            if user.loginToken.prefix(3) == "NAK" {
                Image("Stupa")
                    .resizable()
                    .aspectRatio(contentMode: .fit)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .frame(height: 40)
            }
            
            
            HStack(spacing: 0.0) {
                
                Spacer()
                
                Text(text1)
                Text(highlightText)
                    .font(.system(.largeTitle, design: .rounded))
                    .foregroundColor(.accentColor)
                Text(text2)
                
                Spacer()
            }
            .font(.largeTitle)
            .frame(maxWidth: .infinity)
            
            
        }
        .frame(height: 53)
        .background(Color("Offwhite").opacity(1))
        .overlay(
            RoundedRectangle(cornerRadius: 14, style: .continuous)
                .stroke(
                    LinearGradient(
                        gradient: Gradient(stops: [
                                            .init(color: Color(#colorLiteral(red: 0.7791666388511658, green: 0.7791666388511658, blue: 0.7791666388511658, alpha: 0.949999988079071)), location: 0),
                                            .init(color: Color(#colorLiteral(red: 0.7250000238418579, green: 0.7250000238418579, blue: 0.7250000238418579, alpha: 0)), location: 1)]),
                        startPoint: UnitPoint(x: 0.9016393067273221, y: 0.10416647788375455),
                        endPoint: UnitPoint(x: 0.035519096038869824, y: 0.85416653880629)),
                    lineWidth: 0.5
                )
        )
        .clipShape(RoundedRectangle(cornerRadius: 14, style: .continuous))
        .padding(.horizontal, 24)
        .padding(.bottom, 10)
        .onAppear {
            
            firstly {
                userManger.getCurrentUser()
            }.done { userModel in
                self.user = userModel
                
                print("usertoken: \(user.loginToken.prefix(3))")
                
            }.catch { error in
                print(error.localizedDescription)
            }
            
        }
    }
}

struct HeaderView_Previews: PreviewProvider {
    static var previews: some View {
        HeaderView()
    }
}
