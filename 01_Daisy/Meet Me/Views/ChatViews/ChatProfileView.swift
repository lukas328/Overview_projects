//
//  ChatProfileView.swift
//  Meet Me
//
//  Created by Philipp Hemkemeyer on 05.03.21.
//

import SwiftUI

struct ChatProfileView: View {
    
    
    @Binding var showChatProfileView: Bool
    @Binding var event: EventModel
    
    var body: some View {
        
        ZStack {
            
            // MARK: Tab-able Background
//            Color.black.opacity(0.001)
//                .onTapGesture {
//                    withAnimation(.default) {
//                        showChatProfileView.toggle()
//                    }
//                }
            
            // MARK: Profile on top
            YouProfileNView(showYouProfileView: $showChatProfileView, event: $event)
            
        }
    }
}

struct ChatProfileView_Previews: PreviewProvider {
    static var previews: some View {
        ChatProfileView(showChatProfileView: .constant(true), event: .constant(stockEvent))
    }
}

